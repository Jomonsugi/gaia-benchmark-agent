import os
import warnings

# Suppress TensorFlow warnings (from board_to_fen library)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress INFO/WARN/ERROR
# Force TensorFlow to be single-threaded and CPU-only
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
warnings.filterwarnings("ignore", message=".*tf.function retracing.*")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

import pandas as pd

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_tavily import TavilySearch
from langchain_core.tools import Tool, StructuredTool
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import YoutubeLoader
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import pipeline
import chess
import chess.engine
import librosa
import random
import pathlib
import subprocess
import hashlib
import re
import replicate
from board_to_fen.predict import get_fen_from_image_path
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
from typing import Optional
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError




# The WikipediaQueryRun tool defaults to topKResults=3 and maxDocContentLength=4000.
# This provides up to 3 snippets (each up to ~4000 characters) concatenated into one string.
wikipedia_api_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=4000)
wikipedia_search_tool = WikipediaQueryRun(
    name="wikipedia_search",
    description=(
        "Search Wikipedia and return up to 3 relevant snippets "
        "(each up to ~4000 characters)."
    ),
    api_wrapper=wikipedia_api_wrapper,
)
# Initialize Tavily Search Tool
tavily_web_search_tool = TavilySearch(
    max_results=3,
    topic="general",
    description="Search the web for information on a given topic."
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

# Directory for temporary video downloads
VIDEO_DOWNLOAD_DIR = pathlib.Path(__file__).parent / "downloaded_videos"
VIDEO_DOWNLOAD_DIR.mkdir(exist_ok=True)
CLEANUP_VIDEO_DOWNLOADS = os.getenv("CLEANUP_VIDEO_DOWNLOADS", "false").strip().lower() == "true"


def _download_youtube_video(youtube_url: str) -> str:
    """
    Download a YouTube video as mp4 and return the local filepath.
    """
    if not youtube_url or not youtube_url.strip():
        raise ValueError("youtube_url must be provided.")

    normalized_url = youtube_url.strip()
    video_hash = hashlib.md5(normalized_url.encode("utf-8")).hexdigest()
    output_path = VIDEO_DOWNLOAD_DIR / f"{video_hash}.mp4"

    if output_path.exists() and output_path.stat().st_size > 0:
        return str(output_path)

    cmd = [
        "yt-dlp",
        "-f",
        "mp4",
        "-o",
        str(output_path),
        normalized_url,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        raise RuntimeError(f"yt-dlp failed: {stderr or exc}") from exc

    if not output_path.exists():
        raise RuntimeError("Video download failed; mp4 file not found.")

    return str(output_path)


def youtube_video_visual_qa_tool_func(youtube_url: str, question: str) -> str:
    """
    Download a short YouTube video and answer a visual question using VideoLLaMA3-7B via Replicate.
    Requires REPLICATE_API_TOKEN to be set in the environment.
    """
    local_path = None
    try:
        local_path = _download_youtube_video(youtube_url)

        with open(local_path, "rb") as video_file:
            output = replicate.run(
                "lucataco/videollama3-7b:34a1f45f7068f7121a5b47c91f2d7e06c298850767f76f96660450a0a3bd5bbe",
                input={
                    "video": video_file,
                    "prompt": question,
                },
            )

        if isinstance(output, list):
            text = "".join(str(part) for part in output).strip()
        else:
            text = str(output).strip()

        return text or "ERROR: Empty response from VideoLLaMA3."
    except Exception as e:
        return f"ERROR running youtube_video_visual_qa_tool: {e}"
    finally:
        if CLEANUP_VIDEO_DOWNLOADS and local_path:
            try:
                os.remove(local_path)
            except OSError:
                pass


youtube_video_visual_qa_tool = StructuredTool.from_function(
    func=youtube_video_visual_qa_tool_func,
    name="youtube_video_visual_qa_tool",
    description=(
        "Answer a question about the visual content of a short YouTube video."
        "Input the YouTube URL and your question."
    ),
)


# Web page retrieval helpers

DEFAULT_WEB_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _fetch_url_content(url: str) -> tuple[str, list[tuple[str, str]]]:
    try:
        resp = requests.get(url, headers=DEFAULT_WEB_HEADERS, timeout=20)
    except requests.RequestException as exc:
        return (f"Error fetching {url}: {exc}", [])

    if resp.status_code == 403:
        return (
            "Access blocked (HTTP 403). The site may require solving a CAPTCHA, dismissing a pop-up, "
            "or manual access. As a next step, call interactive_web_browse with this same URL to try a real "
            "browser, or use tavily_web_search_tool with the article title/DOI or authors to find an alternate source.",
            [],
        )

    resp.raise_for_status()

    possible_captcha = resp.text.lower()
    if "captcha" in possible_captcha or "cloudflare" in possible_captcha:
        return (
            "Encountered a CAPTCHA, bot check, or blocking pop-up on this page; static fetching is insufficient. "
            "As a next step, call interactive_web_browse with this same URL to try a real browser, or use "
            "tavily_web_search_tool with the article title/DOI or authors to locate a mirror (e.g. arXiv, ADS, "
            "publisher PDF).",
            [],
        )

    content_type = (resp.headers.get("Content-Type") or "").lower()
    if "text/html" in content_type or content_type.startswith("text/"):
        soup = BeautifulSoup(resp.text, "html.parser")
        for element in soup(["script", "style", "noscript"]):
            element.decompose()
        text = soup.get_text(separator="\n")
        links: list[tuple[str, str]] = []
        for anchor in soup.find_all("a", href=True):
            href = anchor.get("href")
            if not href:
                continue
            href = urljoin(url, href)
            if href.startswith("javascript:"):
                continue
            link_text = anchor.get_text(strip=True) or href
            links.append((link_text, href))
        return text, links
    if "pdf" in content_type or url.lower().endswith(".pdf"):
        return (
            "Encountered a PDF document. Download and process separately if additional tooling is available.",
            [("PDF document", url)],
        )
    return (f"Unsupported content type: {content_type or 'unknown'}", [])


def _format_page_response(
    url: str,
    page_token: int,
    chunk: str,
    total_length: int,
    chunk_chars: int,
    links: list[tuple[str, str]],
) -> str:
    lines = [
        f"Fetched URL: {url}",
        f"Chunk size: {chunk_chars} characters. Page token: {page_token}",
        f"Text chunk:\n{chunk.strip() or '[No text extracted]'}",
    ]
    if links:
        max_links = 20
        formatted_links = "\n".join(
            f"- [{text}]({href})" for text, href in links[:max_links]
        )
        if len(links) > max_links:
            formatted_links += f"\n...and {len(links) - max_links} more links"
        lines.append("\nOutgoing links:\n" + formatted_links)
    next_token = page_token + 1 if (page_token + 1) * chunk_chars < total_length else None
    lines.append(f"next_page_token: {next_token if next_token is not None else 'END'}")
    return "\n\n".join(lines)


def web_fetch_page(url: str, page_token: int = 0, chunk_chars: int = 4000) -> str:
    """
    Fetch a web page (e.g., a result from Tavily or Wikipedia) and return a readable text chunk plus outgoing links.
    Use page_token to paginate through long pages (each chunk ~4000 characters).
    Input: Absolute URL of the page to fetch (must be HTTP/HTTPS).
    Output: Readable text chunk, plus the list of outgoing links detected on that page.
    """
    text, links = _fetch_url_content(url)
    start = max(page_token, 0) * chunk_chars
    if start >= len(text):
        return (
            f"Fetched URL: {url}\n"
            f"Chunk size: {chunk_chars} characters. Page token: {page_token}\n"
            "Reached the end of the document. next_page_token: END"
        )
    end = start + chunk_chars
    chunk = text[start:end]
    return _format_page_response(url, page_token, chunk, len(text), chunk_chars, links)


web_fetch_page_tool = StructuredTool.from_function(
    func=web_fetch_page,
    name="web_fetch_page",
    description=(
        "Fetch a web page and return readable text plus its outgoing links. "
        "Provide page_token to continue reading long documents."
    ),
)


# Playwright-powered browser mini-agent

def interactive_web_browse(
    url: str,
    max_scrolls: int = 3,
    wait_seconds: float = 1.5,
) -> str:
    """
    Open a web page in a real browser using Playwright, handle dynamic content and common pop-ups,
    scroll the page, and return a large chunk of readable text.

    General usage:
    - Prefer static tools like web_fetch_page first for simple pages.
    - Use this tool when static fetching returns blocked/empty/truncated content (e.g. CAPTCHA, 403)
      or when the page clearly relies on JavaScript and scrolling to reveal information.

    Inputs:
    - url: Absolute HTTP/HTTPS URL of the page to browse.
    - max_scrolls: Maximum number of scroll operations (default 3). Higher values explore more of a long page.
    - wait_seconds: Seconds to wait after each navigation/scroll to allow content to load.

    Output:
    - A readable text block extracted from the rendered page (truncated if very long), prefixed with the final URL.
    """

    if not url or not url.strip():
        return "ERROR: url must be a non-empty HTTP/HTTPS URL."

    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return "ERROR: url must start with http:// or https://"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()

            # Try a strict load state first, then gracefully fall back if the site never goes idle.
            try:
                page.goto(url, wait_until="networkidle", timeout=20000)
            except PlaywrightTimeoutError:
                page.goto(url, wait_until="domcontentloaded", timeout=20000)

            # Attempt to close simple pop-ups / cookie banners by clicking common buttons
            # using role-based selectors and regex on accessible names (Playwright best practice).
            popup_patterns = [
                r"accept(?: all)?",             # "Accept", "Accept all"
                r"i\s*accept",                  # "I accept"
                r"agree",                       # "Agree", "I agree"
                r"ok(?:ay)?",                   # "OK", "Okay"
                r"continue",                    # "Continue"
                r"(?:reject|decline|disagree)", # "Reject", "Reject all", "Decline", "Disagree"
                r"no thanks",                   # "No thanks"
                r"got it",                      # "Got it"
            ]
            for pattern in popup_patterns:
                try:
                    btn = page.get_by_role("button", name=re.compile(pattern, re.IGNORECASE))
                    if btn.is_visible():
                        btn.click(timeout=1000)
                except Exception:
                    continue

            # Scroll the page to load dynamic content using page.evaluate
            for _ in range(max_scrolls):
                try:
                    page.evaluate("window.scrollBy(0, document.documentElement.clientHeight);")
                    page.wait_for_timeout(int(wait_seconds * 1000))
                except Exception:
                    break

            text = page.text_content("body") or ""
            final_url = page.url

            context.close()
            browser.close()

        text = text.strip()
        if not text:
            return (
                f"Browser visited {final_url} but extracted no text. "
                "The page may be heavily scripted, fully blocked by a CAPTCHA, or require login. "
                "Use tavily_web_search_tool with the article title/DOI or authors to search for alternate sources "
                "(e.g. arXiv, ADS, publisher PDF) and read from there."
            )

        max_chars = 12000
        truncated = text[:max_chars]
        return (
            f"Browser visited URL: {final_url}\n\n"
            f"Extracted text (first {max_chars} characters):\n{truncated}"
        )
    except Exception as e:
        return f"ERROR in interactive_web_browse: {e}"


interactive_web_browse_tool = StructuredTool.from_function(
    func=interactive_web_browse,
    name="interactive_web_browse",
    description=(
        "Use a real browser (Playwright) to open a web page, handle dynamic content and simple pop-ups, "
        "scroll the page, and return readable text. Prefer web_fetch_page first; "
        "use this when static fetching is blocked or clearly incomplete (e.g. CAPTCHA, 403, missing content)."
    ),
)


# GAIA benchmark file download
GAIA_REPO_ID = "gaia-benchmark/GAIA"
GAIA_CACHE_DIR = pathlib.Path(__file__).parent / "gaia_cache"
GAIA_CACHE_DIR.mkdir(exist_ok=True)


def download_file_attachment(file_name: str) -> str:
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

def read_excel_file(file_path: str) -> str:
    """
    Read a excel file and return its contents with full decimal precision preserved.
    Use for .xlsx and .xls files.
    
    Input: absolute file path
    Output: file contents of the dataframe in string format
    """
    try:
        df = pd.read_excel(file_path)
        
        # Set pandas display options to preserve high precision
        with pd.option_context('display.precision', 10, 'display.max_columns', None, 'display.width', None):
            df_str = df.to_string(index=False)
        
        return f"Excel file loaded ({len(df)} rows, {len(df.columns)} columns). Full contents:\n\n{df_str}"
    except Exception as e:
        return f"Error reading Excel file: {e}"

read_excel_file_tool = Tool(
    name="ExcelReader",
    func=read_excel_file,
    description=(
        "Read Microsoft Excel files (.xlsx or .xls). Returns the full DataFrame contents as text. "
        "Use this ONCE per file - the output contains all data needed for analysis. "
        "Input: absolute file path. Output: full DataFrame contents as text."
    )
)


def transcribe_audio(file_path: str) -> str:
    """
    Transcribe audio file (.mp3, .wav, .m4a, etc.) to text using Whisper speech-to-text.
    
    Input: absolute file path
    Output: transcribed text
    """
    try:
        path = pathlib.Path(file_path)
        if not path.exists():
            return f"ERROR: Audio file not found at {file_path}"
        
        # Load audio file using librosa (handles MP3 without ffmpeg dependency)
        # Whisper expects 16kHz sample rate
        audio, sr = librosa.load(str(path), sr=16000)
        
        # Load Whisper model via transformers
        # Using 'base' model for good balance of speed and accuracy
        # For better accuracy, can use 'small' or 'medium', but slower
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            device=-1,  # Use CPU (-1) or GPU (0+) if available
            framework="pt"  # Force PyTorch to avoid TF/Keras 3 path
        )
        
        # Pass raw audio array instead of file path (avoids ffmpeg requirement)
        # For audio > 30 seconds, Whisper requires return_timestamps=True for long-form generation
        # extract just the text from the result
        result = pipe(audio, return_timestamps=True)
        
        # Extract text from result
        if isinstance(result, dict):
            transcript = result.get("text", "")
        else:
            # If result is just a string
            transcript = str(result) if result else ""
        
        if not transcript:
            return "ERROR: Transcription produced no text output."
        
        return transcript
        
    except Exception as e:
        return f"ERROR transcribing audio: {e}"


transcribe_audio_tool = Tool(
    name="transcribe_audio",
    func=transcribe_audio,
    description=(
        "Transcribe audio files (.mp3, .wav, .m4a, etc.) to text using speech-to-text. "
        "Use this ONCE per audio file - the output contains the full transcription. "
        "Input: absolute file path. Output: transcribed text."
    )
)

def _get_perspective_from_image_file_path(image_path: str) -> str:
    """
    Determine if the chess board image is from the perspective of the white or black player.
    Returns "white" or "black".
    """
    question = """Determine if this chess board image is from the perspective of the white or black player. Answer in only one word: "white" or "black"."""
    
    perspective = vision_qa(image_path, question)

    if perspective == "white":
        black_view = False
    elif perspective == "black":
        black_view = True
    else:
        return f"ERROR: Could not determine the perspective of the chess board image. Output: {perspective}."

    return black_view

def _get_fen_in_subprocess(image_path_str: str, black_view: bool, black_to_move: bool) -> str:
    """
    Run get_fen_from_image_path in a subprocess to isolate TensorFlow from LangGraph's execution context.
    """
    import sys
    import json
    
    # Create a script that runs in the subprocess
    script = f"""
import sys
import json
from board_to_fen.predict import get_fen_from_image_path

image_path = {repr(image_path_str)}
black_view = {black_view}
black_to_move = {black_to_move}

try:
    fen = get_fen_from_image_path(image_path, black_view=black_view)
    # Ensure full FEN (side to move etc.). If only piece placement returned, append defaults.
    if fen.count(" ") < 5:
        turn_indicator = "b" if black_to_move else "w"
        fen = fen + " " + turn_indicator + " - - 0 1"
    result = {{"success": True, "fen": fen}}
except Exception as e:
    result = {{"success": False, "error": str(e)}}

print(json.dumps(result))
sys.stdout.flush()
"""
    
    try:
        # Run in subprocess with timeout
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60.0,  # 60 second timeout
            cwd=str(pathlib.Path(__file__).parent)
        )
        
        if result.returncode != 0:
            return f"ERROR: Subprocess failed with code {result.returncode}: {result.stderr}"
        
        output = result.stdout.strip()
        data = json.loads(output)
        
        if data.get("success"):
            return data["fen"]
        else:
            return f"ERROR extracting FEN via board_to_fen: {data.get('error', 'Unknown error')}"
            
    except subprocess.TimeoutExpired:
        return "ERROR: FEN extraction timed out after 60 seconds in subprocess"
    except json.JSONDecodeError:
        return f"ERROR: Failed to parse subprocess output: {result.stdout[:100]}"
    except Exception as e:
        return f"ERROR in subprocess execution: {e}"

def get_fen_from_image_file_path(file_path: str, black_to_move: bool = True) -> str:
    """
    Extract a chess position FEN from a board image (.png/.jpg).
    Returns a full FEN with the correct side to move indicator.
    
    Args:
        file_path: Absolute path to the chess board image
        black_to_move: True if black is to move (default), False if white is to move
    """

    image_path = pathlib.Path(file_path)
    if not image_path.exists():
        return f"ERROR: Image not found at {file_path}"

    black_view = _get_perspective_from_image_file_path(image_path)
    
    # Check if black_view is a boolean (it should be)
    if not isinstance(black_view, bool):
        # If it returned an error string, return it
        if isinstance(black_view, str) and black_view.startswith("ERROR"):
            return black_view
        # Unexpected return type - return error
        return f"ERROR: Unexpected return type from perspective detection: {type(black_view)}"

    # Run FEN extraction in subprocess to isolate TensorFlow
    return _get_fen_in_subprocess(str(image_path), black_view, black_to_move)


get_fen_from_image_file_path_tool = StructuredTool.from_function(
    func=get_fen_from_image_file_path,
    name="get_fen_from_image_file_path",
    description=(
        "Extract a chess position FEN string from a board image (.png/.jpg) using board_to_fen. "
        "Input: absolute image path and optionally black_to_move (True for black to move, False for white, defaults to True). "
        "Output: FEN string with correct turn indicator."
    ),
)


# Analyze a FEN with python-chess + Stockfish
def chess_best_move_from_fen(fen: str) -> str:
    """
    Given a FEN string, return the best move for the side to move in algebraic notation (SAN).
    Requires a UCI engine available at STOCKFISH_PATH or in PATH as 'stockfish'.
    """
    try:
        board = chess.Board(fen)
        engine_path = os.getenv("STOCKFISH_PATH") or "stockfish"
        with chess.engine.SimpleEngine.popen_uci(engine_path) as eng:
            result = eng.play(board, chess.engine.Limit(time=2.0))
            move = result.move
            return board.san(move)
    except Exception as e:
        return f"ERROR analyzing FEN: {e}"


chess_best_move_from_fen_tool = Tool(
    name="chess_best_move_from_fen",
    func=chess_best_move_from_fen,
    description=(
        "Analyze a chess position given a FEN string and return the best move in algebraic notation. "
        "Requires STOCKFISH_PATH env or 'stockfish' in PATH. Input: FEN string. Output: SAN move."
    ),
)



# Combined: chess best move from image (FEN extraction + engine)
# Build a pipeline that chains the FEN extractor and Stockfish analyzer
# This allows LangSmith to see individual outputs of each step
def _extract_fen_with_turn(input_dict: dict) -> str:
    """Extract FEN from image with turn indicator - visible in LangSmith"""
    file_path = input_dict["file_path"]
    black_to_move = input_dict.get("black_to_move", True)
    return get_fen_from_image_file_path(file_path, black_to_move=black_to_move)

extract_fen_runnable = RunnableLambda(_extract_fen_with_turn).with_config(
    run_name="extract_fen_from_image", tags=["chess", "fen"]
)

solve_fen_runnable = RunnableLambda(
    chess_best_move_from_fen_tool.func
).with_config(run_name="analyze_fen_with_stockfish", tags=["chess", "stockfish"])

def _chess_pipeline_wrapper(file_path: str, black_to_move: bool = True) -> str:
    """Wrapper that chains FEN extraction and analysis"""
    # Extract FEN (visible in LangSmith)
    # Now safe to use RunnableLambda since TensorFlow runs in subprocess
    input_dict = {"file_path": file_path, "black_to_move": black_to_move}
    fen = extract_fen_runnable.invoke(input_dict)
    if fen.startswith("ERROR"):
        return fen
    # Analyze FEN (visible in LangSmith)
    return solve_fen_runnable.invoke(fen)

chess_best_move_from_image_tool = StructuredTool.from_function(
    func=_chess_pipeline_wrapper,
    name="chess_best_move_from_image",
    description=(
        "Given a chessboard image (.png/.jpg), extract the FEN and return the best move in algebraic notation "
        "using Stockfish. Input: absolute image path and optionally black_to_move (True for black to move, False for white, defaults to True). "
        "Output: SAN move. "
    ),
)

def vision_qa(image_path: str, question: str) -> str:
    """
    Answer a question about an image.
    
    Args:
        image_path: Absolute path to the image file
        question: Question to ask about the image
    
    Returns:
        Model's answer as plain text
    """
    try:
        path = pathlib.Path(image_path)
        if not path.exists():
            return f"ERROR: Image not found at {image_path}"

        with open(path, "rb") as image_file:
            output = replicate.run(
                "google/gemini-2.5-flash",
                input={
                    "images": [image_file],
                    "prompt": question,
                },
            )

        # Join output parts into a single string
        if isinstance(output, list):
            text = "".join(str(part) for part in output).strip()
        else:
            text = str(output).strip()

        return text or "ERROR: Empty response from vision model."
    except Exception as e:
        return f"ERROR vision_qa: {e}"

vision_qa_tool = StructuredTool.from_function(
    func=vision_qa,
    name="vision_qa",
    description=(
        "Answer a question about an image using a multimodal model. "
        "Provide the absolute path to the image file and your question about it."
    ),
)
