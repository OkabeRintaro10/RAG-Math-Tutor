"""RAG Pipeline with Inngest for PDF ingestion and querying.

This module implements a RAG (Retrieval-Augmented Generation) system using:
- FastAPI for the web server
- Inngest for serverless function orchestration
- LangGraph for workflow management
- Guardrails for input/output validation
"""

from __future__ import annotations

import datetime
import os
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from guardrails.errors import ValidationError
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

import inngest
import inngest.fast_api
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

# --- Inngest client setup ---
inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logger,
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)

config_manager = ConfigurationManager()
data_ingestion_config = config_manager.get_data_ingestion_config()
qdrant_config = config_manager.get_data_storing_params()

data_loader = DataLoader(config=data_ingestion_config)
qdrant_storage = QdrantStorage(config=qdrant_config)

input_guard = InputGuard()
output_guard = OutputGuard()

model_name = config_manager.config.models[0].parameters.model
base_url = config_manager.config.models[0].parameters.base_url
api_key = os.getenv("OPENROUTER_API_KEY")


def _make_ids(source_id: str, count: int) -> list[str]:
    """Create stable UUIDs for vector upsert operations.

    Args:
        source_id: Unique identifier for the source document
        count: Number of IDs to generate

    Returns:
        List of UUID strings
    """
    return [
        str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(count)
    ]


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(limit=2, period=datetime.timedelta(minutes=1)),
    rate_limit=inngest.RateLimit(
        limit=1,
        period=datetime.timedelta(hours=4),
        key="event.data.source_id",
    ),
)
async def rag_ingest_pdf(ctx: inngest.Context) -> dict[str, Any]:
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
        chunks = data_loader.load_and_chunk_pdf(filename=pdf_path)
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

        vecs = data_loader.embed_texts(texts=chunks)
        ids = _make_ids(source_id=source_id, count=len(chunks))
        payloads = [
            {"source": source_id, "text": chunks[i]} for i in range(len(chunks))
        ]

        logger.info("Upserting to Qdrant: %d vectors", len(vecs))
        qdrant_storage.upsert(ids=ids, vectors=vecs, payloads=payloads)
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


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
)
async def rag_query_pdf_ai(ctx: inngest.Context) -> dict[str, Any]:
    """Answer questions using RAG pipeline with LangGraph workflow.

    Workflow:
    1. Validate question is math-related
    2. Retrieve relevant documents from vector store
    3. Decide whether to use KB or web search
    4. Generate answer using LLM with guardrails

    Args:
        ctx: Inngest context containing event data

    Returns:
        Dictionary with answer and source documents

    Raises:
        HTTPException: If no question provided in event data
    """
    logger.info("Starting RAG query function")

    def validate_question(state: GraphState) -> dict[str, Any]:
        """Validate that the question is math-related.

        LOGIC FIX: Proper validation using the validate() method.
        Note: We only return fields we want to UPDATE in the state.
        """
        logger.info("Running validate_question node")
        question = state.question
        try:
            # FIX: Use validate() method instead of parse()
            input_guard.validate(text_to_validate=question)
            logger.info("Input validation passed.")
            # Only return is_valid - other fields already have defaults
            return {"is_valid": True}
        except ValidationError as e:
            logger.error("Input validation failed: %s", e)
            return {
                "is_valid": False,
                "generation": "I can only answer math-related questions.",
            }

    def retrieve(state: GraphState) -> dict[str, Any]:
        """Retrieve relevant documents from vector store.

        LOGIC FIX: Better handling of search results and relevance check.
        """
        question = state.question
        logger.info("Retrieving documents for question: %s", question[:50])

        query_vec = data_loader.embed_query(question)
        found = qdrant_storage.search(query_vec, 5)

        # FIX: Handle None case and extract documents properly
        if not found:
            logger.warning("No search results returned from Qdrant")
            return {"documents": [], "is_kb_relevant": False}

        documents = found.get("contexts", [])

        # FIX: Check both existence and non-empty documents
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
        """Decide whether to use web search or generate from KB.

        LOGIC FIX: Safer default handling.
        """
        is_kb_relevant = state.is_kb_relevant
        logger.debug("should_web_search check: is_kb_relevant=%s", is_kb_relevant)

        # FIX: More explicit logic
        if is_kb_relevant:
            return "generate"
        return "web_search"

    def generate(state: GraphState) -> dict[str, Any]:
        """Generate answer using LLM with output guardrails.

        LOGIC FIX: Better error handling and output validation.
        """
        logger.debug("Running generate node")
        question = state.question
        documents = state.documents

        # FIX: Filter out empty documents
        valid_docs = [doc for doc in documents if doc and doc.strip()]
        context_str = "\n\n".join(valid_docs) if valid_docs else ""

        # FIX: Improved prompt with clearer instructions
        if context_str:
            prompt = (
                "You are a helpful math assistant. Use the following context to "
                "answer the question accurately.\n"
                "If the context doesn't contain enough information, say "
                '"I don\'t have enough information to answer this question."\n\n'
                f"Context:\n{context_str}\n\n"
                f"Question:\n{question}\n\n"
                "Answer:\n"
            )
        else:
            prompt = (
                "You are a helpful math assistant. Answer the following question.\n"
                "If you don't know the answer, say "
                '"I don\'t have enough information to answer this question."\n\n'
                f"Question:\n{question}\n\n"
                "Answer:\n"
            )

        try:
            llm = ChatOpenAI(model=model_name, base_url=base_url, api_key=api_key)
            response = llm.invoke(prompt)
            content = getattr(response, "content", str(response))

            # FIX: Use validate() method for output guard
            output_guard.validate(text_to_validate=content)
            logger.info("LLM generation complete and validated (len=%d)", len(content))
            return {"generation": content}

        except ValidationError as ve:
            logger.warning("Output Guardrail Failed: %s", ve)
            # FIX: More informative error message
            return {
                "generation": (
                    "I apologize, but I couldn't generate a reliable answer "
                    "to your question based on the available information."
                )
            }
        except Exception as exc:
            logger.exception("LLM generation failed: %s", exc)
            return {
                "generation": (
                    "An error occurred while generating the answer. Please try again."
                )
            }

    # Build LangGraph workflow
    workflow = StateGraph(GraphState)

    workflow.add_node("validate_question", validate_question)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node(
        "web_search",
        lambda state: {
            "generation": (
                "Web search functionality is under development. "
                "Please try a question that can be answered from the knowledge base."
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
        {"web_search": "web_search", "generate": "generate"},
    )
    workflow.add_edge("web_search", END)
    workflow.add_edge("generate", END)

    app_graph = workflow.compile()

    # Run the graph
    question = ctx.event.data.get("question")
    if not question:
        logger.error("No question provided in event data")
        raise HTTPException(
            status_code=400,
            detail="Missing question in event data",
        )

    # FIX: Initialize GraphState with required fields
    # LangGraph will handle merging updates from nodes
    inputs = {
        "question": question,
        "is_kb_relevant": False,  # Required field with default
    }

    try:
        final_state = await app_graph.ainvoke(inputs)
    except AttributeError:
        logger.warning("Async invoke not available on graph; using sync invoke")
        import asyncio

        loop = asyncio.get_running_loop()
        final_state = await loop.run_in_executor(None, lambda: app_graph.invoke(inputs))

    # FIX: Better default handling
    generation = final_state.get(
        "generation",
        "I was unable to process your question. Please try again.",
    )
    sources = final_state.get("documents", [])

    return {"answer": generation, "sources": sources}


# FastAPI app setup
app = FastAPI(title="RAG Inngest Server")
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])
