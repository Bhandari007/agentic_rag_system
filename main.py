from src.orchestration import DocumentService
from src.database.qdrant_service import QdrantService
from src.embedding.embed_service import EmbeddingModel
from src.data.data_loader import PDFLoader


if __name__ == "__main__":
    # Initialize core services
    embed_service = EmbeddingModel()
    qdrant_service = QdrantService(
        collection_name="test_collection", embed_service=embed_service
    )

    document_service = DocumentService(
        qdrant_service=qdrant_service,
    )

    # Load a document and store it
    loader = PDFLoader(file_path="data/raw/mycv_2025.pdf")
    metadata = {"title": "Sample Document", "author": "Pawan Sapkota Sharma"}

    doc_id, num_chunks = document_service.store_document(loader, metadata)
    print(f"[INFO] Stored document '{doc_id}' with {num_chunks} chunks.")

    # Search
    results = document_service.search_document(
        query="what is the name of the candiate in the cv", top_k=3
    )
    print("\n[INFO] Search Results:")
    for r in results:
        print(f"ID: {r.id}, Score: {r.score}, Text: {r.payload['text']}")
