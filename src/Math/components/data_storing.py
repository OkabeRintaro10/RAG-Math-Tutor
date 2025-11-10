"""Qdrant vector storage with relevance scoring support.

This module provides vector storage operations with similarity scoring
to enable relevance thresholding in the RAG pipeline.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from src.Math.entity.config_entity import DataStoring
from src.Math import logger


class QdrantStorage:
    """Qdrant vector database client with relevance scoring."""

    def __init__(self, config: DataStoring, url="http://localhost:6333"):
        """Initialize Qdrant client and ensure collection exists.

        Args:
            config: Configuration with collection name and vector dimensions
            url: Qdrant server URL
        """
        self.config = config
        self.client = QdrantClient(url=url, timeout=30)

        # Create collection if it doesn't exist
        if not self.client.collection_exists(self.config.collection_name):
            logger.info(f"Creating collection: {self.config.collection_name}")
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.dimension, distance=Distance.COSINE
                ),
            )
            logger.info(f" Collection created successfully")
        else:
            logger.info(f"Using existing collection: {self.config.collection_name}")

    def upsert(
        self, ids: list[str], vectors: list[list[float]], payloads: list[dict]
    ) -> None:
        """Upsert vectors into the collection.

        Args:
            ids: List of unique IDs for each vector
            vectors: List of embedding vectors
            payloads: List of metadata dictionaries for each vector
        """
        if len(ids) != len(vectors) != len(payloads):
            raise ValueError("ids, vectors, and payloads must have the same length")

        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]

        self.client.upsert(collection_name=self.config.collection_name, points=points)

        logger.info(f" Upserted {len(points)} vectors to {self.config.collection_name}")

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> dict[str, list]:
        """Search for similar vectors with optional score filtering.

        Args:
            query_vector: Query embedding vector
            top_k: Maximum number of results to return
            score_threshold: Optional minimum similarity score (0.0-1.0)

        Returns:
            Dictionary with 'contexts', 'sources', and 'scores' keys
        """
        try:
            # Perform search with optional score threshold
            results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query_vector,
                with_payload=True,
                limit=top_k,
                score_threshold=score_threshold,  # Qdrant native filtering
            )

            if not results:
                logger.info("No search results found")
                return {"contexts": [], "sources": [], "scores": []}

            # Extract data from results
            contexts = []
            sources = set()
            scores = []

            for result in results:
                # Get payload safely
                payload = getattr(result, "payload", None) or {}
                text = payload.get("text", "")
                source = payload.get("source", "")
                score = getattr(result, "score", 0.0)

                if text:
                    contexts.append(text)
                    scores.append(score)
                    if source:
                        sources.add(source)

            top_score = scores[0] if scores else 0.0
            logger.info(f"Found {len(contexts)} results.Top score: {top_score:.3f}")

            return {"contexts": contexts, "sources": list(sources), "scores": scores}

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return {"contexts": [], "sources": [], "scores": []}

    def delete_by_source(self, source_id: str) -> bool:
        """Delete all vectors from a specific source.

        Useful for removing outdated documents from the knowledge base.

        Args:
            source_id: The source identifier to delete

        Returns:
            True if deletion was successful
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # Delete points matching the source
            self.client.delete(
                collection_name=self.config.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="source", match=MatchValue(value=source_id))
                    ]
                ),
            )

            logger.info(f"Deleted vectors from source: {source_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete source {source_id}: {e}")
            return False

    def get_collection_info(self) -> dict:
        """Get information about the collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            info = self.client.get_collection(self.config.collection_name)

            return {
                "name": self.config.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
