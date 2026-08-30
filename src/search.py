import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "chroma_db"

# MUST match the model used in vector_store.py
MODEL_NAME = "multi-qa-mpnet-base-dot-v1"

COLLECTION_NAME = "multi_rag_documents"


def search_documents(query, top_k=5):

    print("=" * 70)
    print("MULTI-DOCUMENT RAG - SEMANTIC SEARCH")
    print("=" * 70)

    print(f"\nQuery: {query}")

    # Load the SAME embedding model used for documents
    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    # Convert query into a normalized embedding
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # Connect to ChromaDB
    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    # Search the vector database
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    print("\n" + "=" * 70)
    print(f"TOP {top_k} RETRIEVED CHUNKS")
    print("=" * 70)

    for i in range(len(results["documents"][0])):

        document_text = results["documents"][0][i]
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        print(f"\nRESULT {i + 1}")
        print("-" * 70)

        print(
            f"Document : "
            f"{metadata['document']}"
        )

        print(
            f"Page     : "
            f"{metadata['page']}"
        )

        print(
            f"Chunk ID : "
            f"{metadata['chunk_id']}"
        )

        print(
            f"Distance : "
            f"{distance:.4f}"
        )

        print("\nRetrieved Text:")
        print(document_text[:700])
        print()


if __name__ == "__main__":

    query = input(
        "\nEnter your question: "
    )

    search_documents(query)