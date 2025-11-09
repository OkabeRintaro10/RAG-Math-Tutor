"""RAG Pipeline with Inngest for PDF ingestion and querying.

This module implements a RAG (Retrieval-Augmented Generation) system using:
- FastAPI for the web server
- Inngest for serverless function orchestration
- LangGraph for workflow management
- Guardrails for input/output validation
- HITL (Human-in-the-Loop) feedback collection for DSPy optimization
"""

from __future__ import annotations

import asyncio
import datetime
import os
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import HTTPException
from guardrails.errors import ValidationError
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

import inngest
from guard.guardrails import InputGuard, OutputGuard
from src.Math import logger
from src.Math.components.data_ingestion import DataLoader
from src.Math.components.data_storing import QdrantStorage
from src.Math.config.configuration import ConfigurationManager
from src.Math.entity.config_entity import (
    GraphState,
    RAGChunkAndSrc,
    RAGUpsertResult,
)

# ---- Global Setup -----
load_dotenv()


class RAGPipeline:
    """RAG Pipeline for PDF ingestion and querying with LangGraph workflow."""

    FEEDBACK_FILE = "feedback.jsonl"

    # Configuration for feedback storage # JSONL format for easy processing with DSPy

    def __init__(self, config: ConfigurationManager | None = None):
        """Initialize RAG Pipeline with configuration.

        Args:
            config: Configuration manager instance. If None, creates new instance.
        """
        self.config_manager = config or ConfigurationManager()
        self.data_ingestion_config = self.config_manager.get_data_ingestion_config()
        self.qdrant_config = self.config_manager.get_data_storing_params()

        self.data_loader = DataLoader(config=self.data_ingestion_config)
        self.qdrant_storage = QdrantStorage(config=self.qdrant_config)

        self.input_guard = InputGuard()
        self.output_guard = OutputGuard()

        self.model_name = self.config_manager.config.models[0].parameters.model
        self.base_url = self.config_manager.config.models[0].parameters.base_url
        self.api_key = os.getenv("OPENROUTER_API_KEY")

        # Initialize Inngest client
        self.inngest_client = inngest.Inngest(
            app_id="rag_app",
            logger=logger,
            is_production=False,
            serializer=inngest.PydanticSerializer(),
        )

        # Build and compile the workflow
        self.app_graph = self._build_workflow()

        # Register Inngest functions after all methods are defined
        self._register_inngest_functions()
        self.SUMMARY_THRESHOLD = 102400

    def _make_ids(self, source_id: str, count: int) -> list[str]:
        """Create stable UUIDs for vector upsert operations.

        Args:
            source_id: Unique identifier for the source document
            count: Number of IDs to generate

        Returns:
            List of UUID strings
        """
        return [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
            for i in range(count)
        ]

    def _register_inngest_functions(self) -> None:
        """Register Inngest functions with the client."""
        # Decorate the handler methods and store as attributes
        self._rag_ingest_pdf = self.inngest_client.create_function(
            fn_id="RAG: Ingest PDF",
            trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
            throttle=inngest.Throttle(limit=2, period=datetime.timedelta(minutes=1)),
            rate_limit=inngest.RateLimit(
                limit=1,
                period=datetime.timedelta(hours=4),
                key="event.data.source_id",
            ),
        )(self.rag_ingest_pdf_handler)

        self._rag_query_pdf_ai = self.inngest_client.create_function(
            fn_id="RAG: Query PDF",
            trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
        )(self.rag_query_pdf_ai_handler)

    async def rag_ingest_pdf_handler(self, ctx: inngest.Context) -> dict[str, Any]:
        """Load, chunk, embed and upsert a PDF into Qdrant.

        Args:
            ctx: Inngest context containing event data

        Returns:
            Dictionary with ingestion results

        Raises:
            FileNotFoundError: If the PDF file doesn't exist
        """
        logger.info("Starting RAG ingest PDF function")

        def _load(ctx_inner: inngest.Context) -> RAGChunkAndSrc:
            """Load and chunk PDF document."""
            pdf_path_str = ctx_inner.event.data["pdf_path"]
            source_id = ctx_inner.event.data.get("source_id", pdf_path_str)
            pdf_path = Path(pdf_path_str)

            if not pdf_path.exists():
                logger.error("PDF file not found: %s", pdf_path_str)
                raise FileNotFoundError(f"Source PDF not found at {pdf_path_str}")

            logger.info("Loading and chunking PDF: %s", pdf_path)
            chunks = self.data_loader.load_and_chunk_pdf(filename=pdf_path)
            if not isinstance(chunks, list):
                logger.warning("Data loader returned non-list chunks; coercing to list")
                chunks = list(chunks)

            return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

        def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
            """Embed chunks and upsert to Qdrant."""
            chunks = chunks_and_src.chunks
            source_id = chunks_and_src.source_id

            if not chunks:
                logger.info("No chunks to upsert for source: %s", source_id)
                return RAGUpsertResult(ingested=0)

            logger.info("Embedding %d chunks for source %s", len(chunks), source_id)

            vecs = self.data_loader.embed_texts(texts=chunks)
            ids = self._make_ids(source_id=source_id, count=len(chunks))
            payloads = [
                {"source": source_id, "text": chunks[i]} for i in range(len(chunks))
            ]

            logger.info("Upserting to Qdrant: %d vectors", len(vecs))
            self.qdrant_storage.upsert(ids=ids, vectors=vecs, payloads=payloads)
            return RAGUpsertResult(ingested=len(chunks))

        chunks_and_src = await ctx.step.run(
            "load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc
        )
        upsert_result = await ctx.step.run(
            "embed-and-upsert",
            lambda: _upsert(chunks_and_src),
            output_type=RAGUpsertResult,
        )
        logger.info("Ingest completed: %d chunks ingested", upsert_result.ingested)
        return upsert_result.model_dump()

    async def rag_query_pdf_ai_handler(self, ctx: inngest.Context) -> dict[str, Any]:
        """Answer questions using RAG pipeline with LangGraph workflow.

        Args:
            ctx: Inngest context containing event data

        Returns:
            Dictionary with answer and source documents

        Raises:
            HTTPException: If no question provided in event data
        """
        logger.info("Starting RAG query function via Inngest")

        question = ctx.event.data.get("question")
        if not question:
            logger.error("No question provided in event data")
            raise HTTPException(
                status_code=400,
                detail="Missing question in event data",
            )

        inputs = {
            "question": question,
            "is_kb_relevant": False,
        }

        try:
            final_state = await self.app_graph.ainvoke(inputs)
        except AttributeError:
            logger.warning("Async invoke not available on graph; using sync invoke")
            loop = asyncio.get_running_loop()
            final_state = await loop.run_in_executor(
                None, lambda: self.app_graph.invoke(inputs)
            )

        generation = final_state.get(
            "generation",
            "I was unable to process your question. Please try again.",
        )
        sources = final_state.get("documents", [])

        return {"answer": generation, "sources": sources}

    def _build_workflow(self) -> Any:
        """Build and compile the LangGraph workflow for reuse.

        Returns:
            Compiled LangGraph workflow
        """

        def prepare_context(state: GraphState) -> dict[str, Any]:
            """
            Smartly prepares context:
            1. If history is empty, does nothing.
            2. If history is small ( < 80% context), uses it directly.
            3. If history is large ( > 80% context), summarizes it.

            The output is always stored in the 'summary' state key.
            """

            logger.debug("Running prepare_context node")
            history = state.history
            question = state.question
            current_history_tokens = state.history_tokens

            if not history:
                return {"summary": "", "history_tokens": 0}

            logger.info(f"Context threshold set to {self.SUMMARY_THRESHOLD} tokens.")

            if current_history_tokens <= self.SUMMARY_THRESHOLD:
                # --- A: History is small, use it as-is ---
                logger.info("History is below threshold. Using full history.")
                prompt_history = []
                for msg in history:
                    role = "User" if msg["sender"] == "user" else "Assistant"
                    prompt_history.append(f"{role}: {msg['text']}")
                history_str = "\n".join(prompt_history)

                # Pass the summary and the *unchanged* token count
                return {
                    "summary": history_str,
                    "history_tokens": current_history_tokens,
                }
            else:
                # --- B: History is large, summarize and RESET token count ---
                logger.info("History is over threshold. Summarizing...")
                # (Format history_str for the prompt)
                prompt_history = []
                for msg in history:
                    role = "User" if msg["sender"] == "user" else "Assistant"
                    prompt_history.append(f"{role}: {msg['text']}")
                history_str = "\n".join(prompt_history)

                logger.info("History is over threshold. Summarizing...")
                summary_prompt = (
                    "You are a helpful summarization assistant. "
                    "Condense the following conversation into a short paragraph. Maintain Key Topics from the conversation especially any formulas or transformations"
                    "Focus on the main topics and any information relevant to the user's *new* question.\n\n"
                    "--- CONVERSATION ---\n"
                    f"{history_str}\n\n"
                    "--- USER'S NEW QUESTION ---\n"
                    f"{question}\n\n"
                    "--- CONCISE SUMMARY ---\n"
                )
                try:
                    summarizer_llm = ChatOpenAI(
                        model=self.model_name,
                        base_url=self.base_url,
                        api_key=self.api_key,
                    )
                    response = summarizer_llm.invoke(summary_prompt)

                    new_summary = getattr(response, "content", str(response))
                    # Get the *exact* cost of the summary
                    new_token_count = response.response_metadata["token_usage"][
                        "total_tokens"
                    ]

                    logger.info(
                        f"Summarization complete. New token count: {new_token_count}"
                    )
                    # Return the new summary and the NEW token count
                    return {"summary": new_summary, "history_tokens": new_token_count}

                except Exception as e:
                    logger.error(f"Summarization failed: {e}")
                    # Fail gracefully: return empty summary and reset tokens
                    return {"summary": "", "history_tokens": 0}

        def validate_question(state: GraphState) -> dict[str, Any]:
            """Validate that the question is math-related."""
            logger.info("Running validate_question node")
            question = state.question
            try:
                self.input_guard.validate(text_to_validate=question)
                logger.info("Input validation passed.")
                return {"is_valid": True}
            except ValidationError as e:
                logger.error("Input validation failed: %s", e)
                return {
                    "is_valid": False,
                    "generation": "I can only answer math-related questions.",
                }

        def retrieve(state: GraphState) -> dict[str, Any]:
            """Retrieve relevant documents from vector store."""
            question = state.question
            logger.info("Retrieving documents for question: %s", question[:50])

            query_vec = self.data_loader.embed_query(question)
            found = self.qdrant_storage.search(query_vec, 5)

            if not found:
                logger.warning("No search results returned from Qdrant")
                return {"documents": [], "is_kb_relevant": False}

            documents = found.get("contexts", [])
            is_kb_relevant = bool(documents and any(doc.strip() for doc in documents))

            logger.info(
                "KB relevance: %s (found %d docs)",
                is_kb_relevant,
                len(documents),
            )

            return {
                "documents": documents,
                "is_kb_relevant": is_kb_relevant,
            }

        def should_web_search(state: GraphState) -> str:
            """Decide whether to use web search or generate from KB."""
            is_kb_relevant = state.is_kb_relevant
            logger.debug("should_web_search check: is_kb_relevant=%s", is_kb_relevant)

            if is_kb_relevant:
                return "generate"
            return "web_search"

        def generate(state: GraphState) -> dict[str, Any]:
            """Generate answer using LLM with output guardrails."""
            logger.debug("Running generate node")
            question = state.question
            documents = state.documents
            summary = state.summary
            current_history_tokens = state.history_tokens

            valid_docs = [doc for doc in documents if doc and doc.strip()]
            context_str = "\n\n".join(valid_docs) if valid_docs else ""

            if context_str:
                prompt = (
                    "You are a helpful math assistant. Use the following context to "
                    "answer the question accurately.\n"
                    "If the context doesn't contain enough information, say "
                    '"I don\'t have enough information to answer this question."\n\n'
                    f"Context:\n{context_str}\n\n"
                    f"Coversation History:\n{summary}\n\n"
                    f"Question:\n{question}\n\n"
                    "Answer:\n"
                )
            else:
                prompt = (
                    "You are a helpful math assistant. Answer the following question.\n"
                    "If you don't know the answer, say "
                    '"I don\'t have enough information to answer this question."\n\n'
                    f"Question:\n{question}\n\n"
                    f"Conversation History:\n{summary}\n\n"
                    "Answer:\n"
                )

            try:
                llm = ChatOpenAI(
                    model=self.model_name, base_url=self.base_url, api_key=self.api_key
                )
                response = llm.invoke(prompt)
                generation_cost = response.response_metadata["token_usage"][
                    "total_tokens"
                ]
                # Add it to the running total
                new_total_tokens = current_history_tokens + generation_cost
                content = getattr(response, "content", str(response))

                self.output_guard.validate(text_to_validate=content)
                logger.info(
                    "LLM generation complete and validated (len=%d)", len(content)
                )
                return {"generation": content, "history_tokens": new_total_tokens}

            except ValidationError as ve:
                logger.warning("Output Guardrail Failed: %s", ve)
                return {
                    "generation": (ve),
                    "history_tokens": current_history_tokens,
                }
            except Exception as exc:
                logger.exception("LLM generation failed: %s", exc)
                return {
                    "generation": (
                        "An error occurred while generating the answer. "
                        "Please try again."
                    ),
                    "history_tokens": current_history_tokens,
                }

        workflow = StateGraph(GraphState)

        workflow.add_node("prepare_context", prepare_context)
        workflow.add_node("validate_question", validate_question)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node(
            "web_search",
            lambda state: {
                "generation": (
                    "Web search functionality is under development. "
                    "Please try a question that can be answered from the "
                    "knowledge base."
                )
            },
        )
        workflow.add_node("generate", generate)

        workflow.set_entry_point("validate_question")

        def after_validation(state: GraphState) -> str:
            """Route based on validation result."""
            return "retrieve" if state.is_valid else END

        workflow.add_conditional_edges(
            "validate_question",
            after_validation,
            {END: END, "retrieve": "retrieve"},
        )
        workflow.add_conditional_edges(
            "retrieve",
            should_web_search,
            {"web_search": "web_search", "generate": "prepare_context"},
        )
        workflow.add_edge("web_search", END)
        workflow.add_edge("prepare_context", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    async def query(self, question: str) -> dict[str, Any]:
        """Query the RAG pipeline directly without Inngest.

        Args:
            question: The question to answer

        Returns:
            Dictionary with answer and source documents

        Raises:
            ValueError: If no question provided
        """
        if not question:
            raise ValueError("Question cannot be empty")

        inputs = {
            "question": question,
            "is_kb_relevant": False,
        }

        try:
            final_state = await self.app_graph.ainvoke(inputs)
        except AttributeError:
            loop = asyncio.get_running_loop()
            final_state = await loop.run_in_executor(
                None, lambda: self.app_graph.invoke(inputs)
            )

        generation = final_state.get(
            "generation",
            "I was unable to process your question. Please try again.",
        )
        sources = final_state.get("documents", [])

        return {"answer": generation, "sources": sources}

    def get_inngest_client(self) -> inngest.Inngest:
        """Get the Inngest client for FastAPI integration.

        Returns:
            Configured Inngest client
        """
        return self.inngest_client
