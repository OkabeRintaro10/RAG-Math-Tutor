from __future__ import annotations
import uuid
from typing import Any

from src.Math import logger
from src.Math.entity.config_entity import GraphState


class StoreAndUpdate:
    """Component to store new knowledge back into the Vector DB."""

    def __init__(self, data_loader, qdrant_storage):
        """Initialize with data loader and storage components."""
        self.data_loader = data_loader
        self.qdrant_storage = qdrant_storage

    def _generate_document_id(self, question: str) -> str:
        """Create a stable, unique ID for a new document based on the question."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"qa-{question}"))

    async def store_answer(self, state: GraphState) -> dict[str, Any] | None:
        """
        Takes a generated answer, formats it as a Q&A document, embeds it,
        and upserts it into the Qdrant vector database.
        """
        question = state.question
        generation = state.generation
        if not question or not generation:
            logger.warning(
                "Store node: Missing question or generation in state. Skipping."
            )
            return None

        logger.info("🧠 Storing new Q&A knowledge into Vector DB...")

        # 1. Format the new document
        new_document = f"Question: {question}\n\nAnswer: {generation}"
        source_id = f"self-generated-qa-{uuid.uuid4().hex[:8]}"

        try:
            # 2. Embed the new document
            embedding = self.data_loader.embed_query(
                new_document
            )  # Use query embedding for Q&A

            # 3. Generate a stable ID
            doc_id = self._generate_document_id(question)

            # 4. Prepare payload
            payload = {"source": source_id, "text": new_document}

            # 5. Upsert into Qdrant
            self.qdrant_storage.upsert(
                ids=[doc_id], vectors=[embedding], payloads=[payload]
            )
            logger.info(f"✅ Successfully stored new knowledge with ID: {doc_id}")

        except Exception as e:
            logger.error(
                f"❌ Failed to store new knowledge in Vector DB: {e}", exc_info=True
            )

        # This is a terminal node, so it doesn't need to return anything to the state
        return None
