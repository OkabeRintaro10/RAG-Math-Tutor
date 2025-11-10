from typing import Any

from src.Math.components.data_ingestion import DataLoader
from src.Math.components.data_storing import QdrantStorage
from src.Math.config.configuration import ConfigurationManager
from src.Math import logger
from src.Math.entity.config_entity import GraphState


class DataRetrieve:
    def __init__(
        self,
        config: ConfigurationManager,
        DataLoader: DataLoader,
        QdrantStorage: QdrantStorage,
        RELEVANCE_THRESHOLD: float,
    ):
        self.config_manager = config
        self.data_loader = DataLoader
        self.qdrant_storage = QdrantStorage
        self.RELEVANCE_THRESHOLD = RELEVANCE_THRESHOLD

    def retrieve(self, state: GraphState) -> dict[str, Any]:
        """Retrieve relevant documents from vector store with relevance scoring.

        Returns:
            Dictionary with documents and is_kb_relevant flag
        """
        question = state.question
        logger.info(f"Retrieving documents for: '{question[:50]}...'")

        # Generate query embedding
        query_vec = self.data_loader.embed_query(question)

        # Search with relevance threshold
        found = self.qdrant_storage.search(
            query_vec, top_k=5, score_threshold=self.RELEVANCE_THRESHOLD
        )

        if not found or not found.get("contexts"):
            logger.warning("No search results returned from Qdrant")
            return {"documents": [], "is_kb_relevant": False}

        documents = found.get("contexts", [])
        scores = found.get("scores", [])

        if not scores:
            # Fallback if no scores returned
            logger.warning("No scores returned. Using simple relevance check.")
            is_kb_relevant = bool(documents and any(doc.strip() for doc in documents))
        else:
            # Check top score against threshold
            top_score = scores[0] if scores else 0
            logger.info(f"Top document similarity score: {top_score:.3f}")

            if top_score >= self.RELEVANCE_THRESHOLD:
                is_kb_relevant = True
                # Filter to only include documents above threshold
                filtered_docs = [
                    doc
                    for doc, score in zip(documents, scores)
                    if score >= self.RELEVANCE_THRESHOLD
                ]
                documents = filtered_docs
                logger.info(
                    f"✅ Found {len(documents)} relevant documents (score ≥ {self.RELEVANCE_THRESHOLD})"
                )
            else:
                is_kb_relevant = False
                documents = []
                logger.info(
                    f"❌ Top score {top_score:.3f} below threshold {self.RELEVANCE_THRESHOLD}. KB not relevant."
                )

        return {
            "documents": documents,
            "is_kb_relevant": is_kb_relevant,
        }
