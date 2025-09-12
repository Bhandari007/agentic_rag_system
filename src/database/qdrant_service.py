import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.embedding.embed_service import EmbeddingModel
from langchain.schema import Document


class QdrantService:
    def __init__(
        self, collection_name: str, embed_service: EmbeddingModel, ndim: int = 384
    ):
        self.client = QdrantClient(host="localhost", port=8000)
        self.collection_name = collection_name
        self.embed_service = embed_service
        self.vector_size = ndim

        # Create collection
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=ndim, distance=models.Distance.COSINE
            ),
        )
        print(
            f"[INFO] Qdrant collection '{collection_name}' initialized with {ndim}-dimensional vectors."
        )

    def store_document(self, loader, metadata: dict):
        """
        Full pipeline: split document, embed, and store into Qdrant.
        """
        # Step 1: Load & split document
        chunks: list[Document] = loader.load_and_split()
        document_id = uuid.uuid4()

        # Step 2: Create Qdrant points
        points = []
        for idx, chunk in enumerate(chunks):
            vector = self.embed_service.embed_text(chunk.page_content)
            points.append(
                models.PointStruct(
                    id=idx,
                    vector=vector,
                    payload={
                        "doc_id": document_id,
                        "chunk_index": idx,
                        "text": chunk.page_content,
                    },
                )
            )

        # Step 3: Upsert into Qdrant
        self.client.upsert(collection_name=self.collection_name, points=points)
        print(
            f"[INFO] Stored {len(chunks)} chunks for document {document_id} in Qdrant."
        )

        return document_id, len(chunks)

    def search(self, query: str, top_k: int = 5, doc_id: str = None):
        """
        Search for documents similar to a query.
        """
        query_vector = self.embed_service.embed_text(query)
        filter_payload = (
            {"must": [{"key": "doc_id", "match": {"value": doc_id}}]}
            if doc_id
            else None
        )

        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=models.Filter(**filter_payload) if filter_payload else None,
        )
