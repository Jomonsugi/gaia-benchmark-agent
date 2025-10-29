from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.retrievers import WikipediaRetriever
from langchain_tavily import TavilySearch
from langchain_core.tools import Tool
from langchain_community.document_loaders import YoutubeLoader
from huggingface_hub import hf_hub_download, list_repo_files
import random
import os
import pathlib
import mimetypes


# Initialize the DuckDuckGo search tool
basic_search_tool = DuckDuckGoSearchRun()
retriever = WikipediaRetriever()

wikipedia_search_tool = Tool(
    name="wikipedia_search",
    func=retriever.invoke,
    description="Search Wikipedia for information on a given topic."
)
# Initialize Tavily Search Tool
tavily_search_tool = TavilySearch(
    max_results=3,
    topic="general",
    description="Search the web for information on a given topic."
)


def get_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""
    # Dummy weather data
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20}
    ]
    # Randomly select a weather condition
    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"

# Initialize the tool
weather_info_tool = Tool(
    name="get_weather_info",
    func=get_weather_info,
    description="Fetches dummy weather information for a given location."
)


def fetch_youtube_transcript(url: str) -> str:
    """Returns the transcript text for a YouTube video URL.

    Expects a standard YouTube watch URL, e.g. https://www.youtube.com/watch?v=...
    """
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
    docs = loader.load()
    # Concatenate all page contents into a single string
    return "\n".join(doc.page_content for doc in docs if getattr(doc, "page_content", None))


youtube_transcript_tool = Tool(
    name="youtube_transcript",
    func=fetch_youtube_transcript,
    description=(
        "Fetch the transcript text from a YouTube video URL. "
        "Input should be the full YouTube URL."
    ),
)


# GAIA benchmark file download
GAIA_REPO_ID = "gaia-benchmark/GAIA"
GAIA_CACHE_DIR = pathlib.Path(__file__).parent / "gaia_cache"
GAIA_CACHE_DIR.mkdir(exist_ok=True)


def download_gaia_attachment(file_name: str) -> str:
    """
    Download a GAIA benchmark attachment file by its file name.
    Returns the absolute local path to the downloaded file.
    
    Input: bare file name like 'abc123.mp3' or 'xyz.py'
    Output: absolute local file path
    """
    fn = (file_name or "").strip()
    if not fn:
        return "ERROR: file_name is empty."
    
    try:
        # Find the file in the GAIA repo (usually under year/split/attachments/)
        all_files = list_repo_files(repo_id=GAIA_REPO_ID, repo_type="dataset")
        matches = [p for p in all_files if os.path.basename(p) == fn]
        
        # Prefer paths with /attachments/ if multiple matches
        matches.sort(key=lambda p: ("/attachments/" not in p, p))
        
        if not matches:
            return f"ERROR: File '{fn}' not found in {GAIA_REPO_ID}."
        
        repo_path = matches[0]
        
        # Download to local cache
        local_path = hf_hub_download(
            repo_id=GAIA_REPO_ID,
            filename=repo_path,
            repo_type="dataset",
            cache_dir=str(GAIA_CACHE_DIR),
        )
        
        return local_path
        
    except Exception as e:
        return f"ERROR downloading GAIA file: {e}"


download_gaia_attachment_tool = Tool(
    name="download_gaia_attachment",
    func=download_gaia_attachment,
    description=(
        "Download a GAIA benchmark attachment file by its file name (e.g., 'abc123.mp3', 'xyz.py'). "
        "Returns the absolute local file path. Use this when a question mentions an attached file."
    ),
)


def read_text_file(file_path: str) -> str:
    """
    Read a local text file and return its contents.
    Use for .py, .txt, .md, .json, .csv files.
    
    Input: absolute file path
    Output: file contents as text
    """
    try:
        path = pathlib.Path(file_path)
        if not path.exists():
            return f"ERROR: File not found at {file_path}"
        
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        
        return content
    except Exception as e:
        return f"ERROR reading file: {e}"


read_text_file_tool = Tool(
    name="read_text_file",
    func=read_text_file,
    description=(
        "Read the contents of a local text file (.py, .txt, .md, .json, .csv). "
        "Input: absolute file path. Output: file contents as text."
    ),
)