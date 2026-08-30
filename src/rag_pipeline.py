import chromadb
from sentence_transformers import SentenceTransformer
import requests


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "multi_rag_documents"

# Must be EXACTLY the same model used in vector_store.py
EMBEDDING_MODEL = "multi-qa-mpnet-base-dot-v1"

# Local Ollama model
OLLAMA_MODEL = "llama3.2:3b"

OLLAMA_URL = "http://localhost:11434/api/generate"

TOP_K = 5


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 70)
print("MULTI-DOCUMENT RAG - GROUNDED ANSWER GENERATION")
print("=" * 70)

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print(
    f"Embedding model loaded: "
    f"{EMBEDDING_MODEL}"
)


# ============================================================
# CONNECT TO CHROMADB
# ============================================================

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print(
    f"Vector database loaded: "
    f"{collection.count()} vectors"
)


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(query, top_k=TOP_K):

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieved = []

    for i in range(
        len(results["documents"][0])
    ):

        retrieved.append({
            "text": results["documents"][0][i],
            "document": results["metadatas"][0][i]["document"],
            "page": results["metadatas"][0][i]["page"],
            "chunk_id": results["metadatas"][0][i]["chunk_id"],
            "distance": results["distances"][0][i]
        })

    return retrieved


# ============================================================
# BUILD GROUNDED PROMPT
# ============================================================

def build_prompt(query, retrieved_documents):

    context_parts = []

    for i, doc in enumerate(
        retrieved_documents,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {i}
Document: {doc['document']}
Page: {doc['page']}
Chunk ID: {doc['chunk_id']}

Content:
{doc['text']}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a document-grounded AI assistant.

You must answer the user's question using ONLY the
information contained in the provided sources.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not invent facts.
3. Do not make assumptions that are not supported
   by the sources.
4. Every important factual claim must have a citation.
5. Use citations in this exact format:

   [Document Name, Page X]

6. If multiple sources support a statement, cite all
   relevant sources.
7. If the provided sources do not contain enough
   information to answer the question, say:

   "The provided documents do not contain enough
   information to answer this question."

8. When sources disagree, explicitly mention the
   disagreement and cite both sources.

USER QUESTION:
{query}

RETRIEVED SOURCES:
{context}

Now provide a concise, well-structured answer with
source citations.
"""

    return prompt


# ============================================================
# GENERATE ANSWER USING OLLAMA
# ============================================================

def generate_answer(prompt):

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data["response"]


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(retrieved_documents):

    print("\n" + "=" * 70)
    print("RETRIEVED SOURCES")
    print("=" * 70)

    for i, doc in enumerate(
        retrieved_documents,
        start=1
    ):

        print(f"\nSOURCE {i}")
        print("-" * 70)

        print(
            f"Document : "
            f"{doc['document']}"
        )

        print(
            f"Page     : "
            f"{doc['page']}"
        )

        print(
            f"Distance : "
            f"{doc['distance']:.4f}"
        )

        print(
            f"Chunk ID : "
            f"{doc['chunk_id']}"
        )


# ============================================================
# MAIN RAG PIPELINE
# ============================================================

def run_rag(query):

    print("\n" + "=" * 70)
    print("RETRIEVING RELEVANT INFORMATION...")
    print("=" * 70)

    retrieved_documents = retrieve_documents(
        query
    )

    display_sources(
        retrieved_documents
    )

    print("\n" + "=" * 70)
    print("GENERATING GROUNDED ANSWER...")
    print("=" * 70)

    prompt = build_prompt(
        query,
        retrieved_documents
    )

    answer = generate_answer(
        prompt
    )

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print("\n" + answer)

    print("\n" + "=" * 70)
    print("END OF RAG RESPONSE")
    print("=" * 70)


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    question = input(
        "\nEnter your question: "
    )

    run_rag(question)