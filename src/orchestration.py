from src.database.qdrant_service import QdrantService


class DocumentService:
    def __init__(
        self,
        qdrant_service: QdrantService,
    ):
        self.qdrant_service = qdrant_service

    def store_document(self, loader, metadata: dict):
        """
        Store a document using the provided loader and metadata.
        """
        document_id, num_chunks = self.qdrant_service.store_document(loader, metadata)

        # Add  metadata to Mongo #-> Todo
        metadata["_id"] = document_id
        metadata["num_chunks"] = num_chunks
        metadata["chunking_strategy"] = {
            "chunk_size": loader.CHUNK_SIZE,
            "chunk_overlap": loader.CHUNK_OVERLAP,
            "add_start_index": True,
        }

        return document_id, num_chunks

    def search_document(self, query: str, top_k: int = 5, doc_id: str = None):
        """
        Search documents in Qdrant and return results.
        """
        return self.qdrant_service.search(query, top_k, doc_id)
