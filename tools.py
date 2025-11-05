import os
import warnings

# Suppress TensorFlow warnings (from board_to_fen library)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress INFO/WARN/ERROR
warnings.filterwarnings("ignore", message=".*tf.function retracing.*")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

import pandas as pd

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.retrievers import WikipediaRetriever
from langchain_tavily import TavilySearch
from langchain_core.tools import Tool
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import YoutubeLoader
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import pipeline
import chess
import chess.engine
import librosa
import random
import pathlib


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

# Create the LangChain tool
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


from board_to_fen.predict import get_fen_from_image_path


def get_fen_from_image_file_path(file_path: str) -> str:
    """
    Extract a chess position FEN from a board image (.png/.jpg) using board_to_fen.
    Returns a full FEN. If side-to-move is missing, defaults to black to move.
    """
    try:
        path = pathlib.Path(file_path)
        if not path.exists():
            return f"ERROR: Image not found at {file_path}"
        fen = get_fen_from_image_path(str(path))
        # Ensure full FEN (side to move etc.). If only piece placement returned, append defaults.
        if fen.count(" ") < 5:
            fen = f"{fen} b - - 0 1"
        return fen
    except Exception as e:
        return f"ERROR extracting FEN via board_to_fen: {e}"


get_fen_from_image_file_path_tool = Tool(
    name="get_fen_from_image_file_path",
    func=get_fen_from_image_file_path,
    description=(
        "Extract a chess position FEN string from a board image (.png/.jpg) using board_to_fen. "
        "Input: absolute image path. Output: FEN string (assumes black to move by default)."
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


# -----------------------------
# Combined: chess best move from image (FEN extraction + engine)
# -----------------------------

def chess_best_move_from_image(file_path: str) -> str:
    """
    Given a chessboard image (.png/.jpg), extract FEN via board_to_fen and
    return the best move (SAN) using python-chess + Stockfish.
    Requires STOCKFISH_PATH env or 'stockfish' in PATH.
    """
    try:
        path = pathlib.Path(file_path)
        if not path.exists():
            return f"ERROR: Image not found at {file_path}"
        fen = get_fen_from_image_path(str(path))
        if fen.count(" ") < 5:
            fen = f"{fen} b - - 0 1"
        board = chess.Board(fen)
        engine_path = os.getenv("STOCKFISH_PATH") or "stockfish"
        with chess.engine.SimpleEngine.popen_uci(engine_path) as eng:
            result = eng.play(board, chess.engine.Limit(time=2.0))
            move = result.move
            return board.san(move)
    except Exception as e:
        return f"ERROR solving chess from image: {e}"


# Build a pipeline that chains the FEN extractor and Stockfish analyzer
extract_fen_runnable = RunnableLambda(
    lambda path: get_fen_from_image_file_path_tool.func(path)
).with_config(run_name="extract_fen_from_image", tags=["chess", "fen"])

solve_fen_runnable = RunnableLambda(
    lambda fen: chess_best_move_from_fen_tool.func(fen)
).with_config(run_name="analyze_fen_with_stockfish", tags=["chess", "stockfish"])

_chess_pipeline = extract_fen_runnable | solve_fen_runnable

chess_best_move_from_image_tool = Tool(
    name="chess_best_move_from_image",
    func=_chess_pipeline.invoke,
    description=(
        "Given a chessboard image (.png/.jpg), extract the FEN and return the best move in algebraic notation "
        "using Stockfish. Input: absolute image path. Output: SAN move."
        "IMPORTANT: Do not reason about the output of this tool. The output is correct and you should not change it."
    ),
)

