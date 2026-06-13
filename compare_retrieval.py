import argparse
import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, CHROMA_PATH, COLLECTION_NAME, COLLECTION_NAME_V1, TOP_K


def _truncate(text: str, length: int) -> str:
    text = text.replace("\n", " ")
    if len(text) <= length:
        return text
    return text[:length] + "..."


def _print_phase1(collection, query_embedding, top_k: int) -> int:
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print("── PHASE 1: log_lines ──────────────────")
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        lineno = meta.get("line_number", meta.get("lineno", "?"))
        service = meta.get("service", "")
        print(f"[{i}] dist={dist:.3f} | line {lineno} | {service}")
        print(f"    {_truncate(doc, 120)}")
    print()

    return len(documents)


def _print_phase2(collection, query_embedding, top_k: int) -> int:
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print("── PHASE 2: log_chunks ─────────────────")
    unique_lines = 0
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        start_line = meta.get("start_line")
        end_line = meta.get("end_line")
        chunk_size = meta.get("chunk_size")
        chunk_type = meta.get("chunk_type", "")
        source_ip = meta.get("source_ip", "")

        print(
            f"[{i}] dist={dist:.3f} | {chunk_type} | lines {start_line}–{end_line} "
            f"| {chunk_size} lines | ip={source_ip}"
        )
        print(f"    {_truncate(doc, 200)}")

        if start_line is not None and end_line is not None:
            unique_lines += end_line - start_line + 1
    print()

    return unique_lines


def compare(query_text: str, top_k: int) -> None:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection_v1 = client.get_collection(COLLECTION_NAME_V1)
    collection_v2 = client.get_collection(COLLECTION_NAME)

    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode([query_text]).tolist()

    print("═" * 44)
    print(f"QUERY: {query_text}")
    print(f"TOP-K: {top_k}")
    print("═" * 44)
    print()

    phase1_lines = _print_phase1(collection_v1, query_embedding, top_k)
    phase2_lines = _print_phase2(collection_v2, query_embedding, top_k)

    print("── SUMMARY ──────────────────────────────")
    print(f"Phase 1: {phase1_lines} results covering {phase1_lines} unique lines")
    print(f"Phase 2: {top_k} results covering {phase2_lines} unique lines")

    if phase1_lines > 0:
        ratio = phase2_lines / phase1_lines
        print(f"Context density: Phase 2 retrieved {ratio:.1f}x more log content")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Phase 1 (log_lines) vs Phase 2 (log_chunks) retrieval for the same query."
    )
    parser.add_argument("query", help="The query text to embed and search with.")
    parser.add_argument("--top-k", type=int, default=TOP_K, help=f"Number of results per collection (default: {TOP_K}).")

    args = parser.parse_args()
    compare(args.query, args.top_k)
