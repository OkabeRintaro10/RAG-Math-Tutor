from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from src.Math.entity.config_entity import DataStoring


class QdrantStorage:
    def __init__(self, config: DataStoring, url="http://localhost:6333"):
        self.config = config
        self.client = QdrantClient(url=url, timeout=30)

        if not self.client.collection_exists(self.config.collection_name):
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.dimension, distance=Distance.COSINE
                ),
            )

    def upsert(self, ids, vectors, payloads):
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(self.config.collection_name, points=points)

    def search(self, query_vector, top_k: int = 5):
        results = self.client.search(
            collection_name=self.config.collection_name,
            query_vector=query_vector,
            with_payload=True,
            limit=top_k,
        )

        sources = set(self.config.sources)
        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                self.config.contexts.append(text)
                sources.add(source)

        return {"contexts": self.config.contexts, "sources": list(self.config.sources)}
