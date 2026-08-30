import os
import sys
import chromadb
from sentence_transformers import SentenceTransformer

# Allow importing ingest.py from the same src folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingest import load_pdfs, chunk_documents


# ChromaDB storage location
CHROMA_PATH = "chroma_db"

# Better model for semantic retrieval
MODEL_NAME = "multi-qa-mpnet-base-dot-v1"


def create_vector_database():

    print("=" * 60)
    print("MULTI-RAG - VECTOR DATABASE CREATION")
    print("=" * 60)

    # Load embedding model
    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Model loaded: {MODEL_NAME}")

    # Load PDFs and create chunks
    documents = load_pdfs()
    chunks = chunk_documents(documents)

    if not chunks:
        print("No chunks available.")
        return

    # Connect to persistent ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete old collection
    try:
        client.delete_collection(
            name="multi_rag_documents"
        )
        print("\nOld collection deleted.")
    except:
        pass

    # Create collection using cosine similarity
    collection = client.create_collection(
        name="multi_rag_documents",
        metadata={
            "hnsw:space": "cosine"
        }
    )

    print("\nGenerating embeddings...")
    print(f"Total chunks to process: {len(chunks)}")

    BATCH_SIZE = 32

    for i in range(0, len(chunks), BATCH_SIZE):

        batch = chunks[i:i + BATCH_SIZE]

        texts = [
            chunk["text"]
            for chunk in batch
        ]

        # Generate normalized embeddings
        embeddings = model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True
        )

        metadatas = [
            {
                "document": chunk["document"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"]
            }
            for chunk in batch
        ]

        ids = [
            chunk["chunk_id"]
            for chunk in batch
        ]

        collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )

        print(
            f"Processed "
            f"{min(i + BATCH_SIZE, len(chunks))}"
            f"/{len(chunks)} chunks"
        )

    print("\n" + "=" * 60)
    print("VECTOR DATABASE CREATED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"Total vectors stored: "
        f"{collection.count()}"
    )


if __name__ == "__main__":
    create_vector_database()