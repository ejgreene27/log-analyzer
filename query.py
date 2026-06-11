import argparse
import ollama
import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, OLLAMA_MODEL, CHROMA_PATH, COLLECTION_NAME, TOP_K


def _build_where(args: argparse.Namespace) -> dict | None:
    filters = []

    if args.level:
        filters.append({"log_level": {"$eq": args.level}})
    if args.service:
        filters.append({"service": {"$eq": args.service}})
    if args.ip:
        filters.append({"source_ip": {"$eq": args.ip}})

    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def _build_header(metadata: dict) -> str:
    parts = [f"lines {metadata['start_line']}–{metadata['end_line']}"]
    parts.append(f"service={metadata['service']}")
    parts.append(f"level={metadata['log_level']}")
    if metadata.get("source_ip"):
        parts.append(f"ip={metadata['source_ip']}")
    parts.append(f"type={metadata['chunk_type']}")
    return f"[{' | '.join(parts)}]"


def query(question: str, args: argparse.Namespace) -> None:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    model = SentenceTransformer(EMBEDDING_MODEL)
    question_embedding = model.encode([question]).tolist()

    where = _build_where(args)

    query_kwargs = {
        "query_embeddings": question_embedding,
        "n_results": TOP_K,
        "include": ["documents", "metadatas"],
    }
    if where is not None:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    retrieved_chunks = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not retrieved_chunks:
        print("No matching chunks found. Try broadening your filters or rephrasing the question.")
        return

    context_block = "\n\n".join(
        f"{_build_header(meta)}\n{doc}"
        for doc, meta in zip(retrieved_chunks, metadatas)
    )

    print("--- Context sent to model ---")
    print(context_block)
    print("--- End context ---")

    prompt = (
        "You are a cybersecurity analyst reviewing log data. Use only the log chunks provided below to answer the question.\n"
        "Each chunk is a group of related log lines. The header above each chunk shows its line range, service, severity\n"
        "level, and source IP. When citing evidence in your answer, reference the line range (e.g. \"lines 66–95\").\n"
        "If the answer cannot be determined from the provided chunks, say so.\n\n"
        f"LOG CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    print(response["message"]["content"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the log analyzer RAG pipeline.")
    parser.add_argument("question", help="The question to ask about the logs.")
    parser.add_argument("--level", choices=["warning", "info"], help="Filter by log level.")
    parser.add_argument("--service", help="Filter by service (exact match).")
    parser.add_argument("--ip", help="Filter by source IP (exact match).")

    args = parser.parse_args()
    query(args.question, args)
