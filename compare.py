import argparse
import time
import ollama
import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, CHROMA_PATH, COLLECTION_NAME, COMPARE_MODELS, TOP_K


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


def compare(question: str, args: argparse.Namespace) -> None:
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

    try:
        available_models = {m["model"] for m in ollama.list()["models"]}
    except Exception as e:
        print(f"Warning: could not fetch model list from Ollama ({e}); will attempt all models anyway.")
        available_models = None

    def _is_available(model_name: str) -> bool:
        if available_models is None:
            return True
        if model_name in available_models:
            return True
        if ":" not in model_name and f"{model_name}:latest" in available_models:
            return True
        return False

    timings: dict[str, float] = {}

    for model_name in COMPARE_MODELS:
        print()
        print(f"══ {model_name} {'═' * max(1, 44 - len(model_name) - 4)}")

        if not _is_available(model_name):
            print(f"Model '{model_name}' not found in `ollama list` — skipping.")
            print("═" * 44)
            continue

        start = time.monotonic()
        try:
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            elapsed = time.monotonic() - start
            print(response["message"]["content"])
            print(f"[{elapsed:.1f}s]")
            timings[model_name] = elapsed
        except ollama.ResponseError as e:
            elapsed = time.monotonic() - start
            print(f"Error calling model '{model_name}': {e}")
        except Exception as e:
            elapsed = time.monotonic() - start
            print(f"Unexpected error calling model '{model_name}': {e}")

        print("═" * 44)

    if timings:
        fastest = min(timings.items(), key=lambda kv: kv[1])
        slowest = max(timings.items(), key=lambda kv: kv[1])
        print()
        print(f"Fastest: {fastest[0]} ({fastest[1]:.1f}s) | Slowest: {slowest[0]} ({slowest[1]:.1f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare RAG answer quality across multiple Ollama models.")
    parser.add_argument("question", help="The question to ask about the logs.")
    parser.add_argument("--level", choices=["warning", "info"], help="Filter by log level.")
    parser.add_argument("--service", help="Filter by service (exact match).")
    parser.add_argument("--ip", help="Filter by source IP (exact match).")

    args = parser.parse_args()
    compare(args.question, args)
