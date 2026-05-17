import os
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "6"))
AUTO_APPROVE_THRESHOLD = int(os.getenv("AUTO_APPROVE_THRESHOLD", "3"))
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))
REPORTS_DIR = os.getenv("REPORTS_DIR", "./reports")
THREAD_ID = os.getenv("THREAD_ID", "research-session-1")
