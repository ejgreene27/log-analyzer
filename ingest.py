import sys
import chromadb
from sentence_transformers import SentenceTransformer
from preprocess import parse_line
from chunker import chunk_lines
from config import (
    EMBEDDING_MODEL,
    CHROMA_PATH,
    COLLECTION_NAME,
    BURST_IP_WINDOW_SECONDS,
    CHUNK_WINDOW_SECONDS,
    CHUNK_MAX_SIZE,
)


def ingest(log_path: str) -> None:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    model = SentenceTransformer(EMBEDDING_MODEL)

    with open(log_path, "r", errors="replace") as f:
        parsed_lines = [
            parse_line(lineno, raw)
            for lineno, raw in enumerate(f, start=1)
            if raw.strip()
        ]

    if not parsed_lines:
        print("No non-empty lines found in log file.")
        return

    print(f"Parsed {len(parsed_lines)} lines from {log_path}")

    chunks = chunk_lines(
        parsed_lines,
        source=log_path,
        burst_window_seconds=BURST_IP_WINDOW_SECONDS,
        time_window_seconds=CHUNK_WINDOW_SECONDS,
        max_chunk_size=CHUNK_MAX_SIZE,
    )

    ip_burst_count = sum(1 for c in chunks if c["chunk_type"] == "ip_burst")
    time_window_count = sum(1 for c in chunks if c["chunk_type"] == "time_window")
    print(f"Chunked into {len(chunks)} chunks ({ip_burst_count} ip_burst, {time_window_count} time_window)")

    print(f"Embedding {len(chunks)} chunks...")
    embeddings = model.encode([c["text"] for c in chunks], show_progress_bar=True).tolist()

    ids = [f"chunk_{c['start_line']}_{c['end_line']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{k: v for k, v in c.items() if k != "text"} for c in chunks]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"Stored {len(chunks)} chunks in collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ingest.py <path/to/logfile>")
        sys.exit(1)
    ingest(sys.argv[1])
