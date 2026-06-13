# Model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3"

# Models to compare in compare.py
COMPARE_MODELS = ["llama3", "llama3.1:8b", "qwen2.5:7b"]

# ChromaDB configuration
CHROMA_PATH = "./chroma_db"

# Phase 1 collection (line-based) — kept for comparison querying
COLLECTION_NAME_V1 = "log_lines"

# Phase 2 collection (chunk-based)
COLLECTION_NAME = "log_chunks"

# Chunking parameters
BURST_IP_WINDOW_SECONDS = 120
CHUNK_WINDOW_SECONDS = 60
CHUNK_MAX_SIZE = 30

# Retrieval configuration
TOP_K = 5
