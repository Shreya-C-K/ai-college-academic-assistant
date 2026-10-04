"""Central configuration for the Academic Assistant."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
RESULTS_DIR = BASE_DIR / "results"

COLLEGE_NAME = "NMAM Institute of Technology"
ASSISTANT_NAME = "NMAMIT Bot"

# --- Embeddings / chunking / retrieval ---
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 3
# FAISS returns squared L2 distance on normalized vectors (= 2 - 2*cosine).
# Lower = more similar. Chunks above this threshold are treated as "not relevant".
MAX_L2_DISTANCE = float(os.getenv("MAX_L2_DISTANCE", "1.30"))

# --- LLM ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# --- Graph ---
MAX_REVIEW_RETRIES = 1

# Official courses (used to map what students say to real course names).
# If you replace the sample data with your own syllabus, update this list.
COURSES = [
    "CS501 - Design and Analysis of Algorithms (DAA)",
    "CS502 - Database Management Systems (DBMS)",
    "CS503 - Operating Systems (OS)",
    "CS504 - Machine Learning (ML)",
    "CS505 - Computer Networks (CN)",
    "CS506 - Generative AI and LLM Applications (GenAI)",
]
# Short names shown in the UI / used in plans
COURSE_NAMES = [
    "Design and Analysis of Algorithms",
    "Database Management Systems",
    "Operating Systems",
    "Machine Learning",
    "Computer Networks",
    "Generative AI and LLM Applications",
]


def required_key_name() -> str:
    return "GOOGLE_API_KEY" if LLM_PROVIDER == "gemini" else "GROQ_API_KEY"
