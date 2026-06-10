import sys
import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, CHROMA_PATH, COLLECTION_NAME


def ingest(log_path: str) -> None:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    model = SentenceTransformer(EMBEDDING_MODEL)

    with open(log_path, "r", errors="replace") as f:
        entries = [
            (lineno, line.rstrip("\n"))
            for lineno, line in enumerate(f, start=1)
            if line.strip()
        ]

    if not entries:
        print("No non-empty lines found in log file.")
        return

    line_numbers, lines = zip(*entries)
    line_numbers, lines = list(line_numbers), list(lines)

    print(f"Embedding {len(lines)} lines from {log_path}...")
    embeddings = model.encode(lines, show_progress_bar=True).tolist()

    ids = [f"line_{n}" for n in line_numbers]
    metadatas = [{"line_number": n, "source": log_path} for n in line_numbers]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=lines,
        metadatas=metadatas,
    )

    print(f"Stored {len(lines)} lines in collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ingest.py <path/to/logfile>")
        sys.exit(1)
    ingest(sys.argv[1])
