import sys
import ollama
import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, OLLAMA_MODEL, CHROMA_PATH, COLLECTION_NAME, TOP_K


def query(question: str) -> None:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    model = SentenceTransformer(EMBEDDING_MODEL)
    question_embedding = model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=TOP_K,
        include=["documents", "metadatas"],
    )

    retrieved_lines = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_block = "\n".join(
        f"[line {m['line_number']}] {doc}"
        for doc, m in zip(retrieved_lines, metadatas)
    )

    prompt = (
        "You are a cybersecurity analyst. "
        "Use only the log lines below to answer the question. "
        "If the answer cannot be determined from the logs, say so.\n\n"
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
    if len(sys.argv) < 2:
        print("Usage: python query.py \"<your question>\"")
        sys.exit(1)
    query(" ".join(sys.argv[1:]))
