from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = PROJECT_ROOT / 'cache'

# CONSTANTS 
BM25_K1 = 1.5