"""RAG Pipeline with Inngest for PDF ingestion and querying.

This module implements a RAG (Retrieval-Augmented Generation) system using:
- FastAPI for the web server
- Inngest for serverless function orchestration
- LangGraph for workflow management
- Guardrails for input/output validation
- HITL (Human-in-the-Loop) feedback collection for DSPy optimization
"""

from __future__ import annotations

import datetime
import json
import os
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
    AskRequest,
    AskResponse,
    FeedbackRequest,
    FeedbackResponse,
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

    This Inngest function uses the globally compiled workflow graph.

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
        final_state = await app_graph.ainvoke(inputs)
    except AttributeError:
        logger.warning("Async invoke not available on graph; using sync invoke")
        import asyncio

        loop = asyncio.get_running_loop()
        final_state = await loop.run_in_executor(None, lambda: app_graph.invoke(inputs))

    generation = final_state.get(
        "generation",
        "I was unable to process your question. Please try again.",
    )
    sources = final_state.get("documents", [])

    return {"answer": generation, "sources": sources}


# Configuration for feedback storage
FEEDBACK_FILE = "feedback.jsonl"  # JSONL format for easy processing with DSPy


# --- Build and compile the LangGraph workflow globally ---
def build_workflow() -> Any:
    """Build and compile the LangGraph workflow for reuse.

    Returns:
        Compiled LangGraph workflow
    """

    def validate_question(state: GraphState) -> dict[str, Any]:
        """Validate that the question is math-related."""
        logger.info("Running validate_question node")
        question = state.question
        try:
            input_guard.validate(text_to_validate=question)
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

        query_vec = data_loader.embed_query(question)
        found = qdrant_storage.search(query_vec, 5)

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

        valid_docs = [doc for doc in documents if doc and doc.strip()]
        context_str = "\n\n".join(valid_docs) if valid_docs else ""

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

            output_guard.validate(text_to_validate=content)
            logger.info("LLM generation complete and validated (len=%d)", len(content))
            return {"generation": content}

        except ValidationError as ve:
            logger.warning("Output Guardrail Failed: %s", ve)
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

    return workflow.compile()


# Compile the workflow globally for reuse
app_graph = build_workflow()


# FastAPI app setup
app = FastAPI(title="RAG Inngest Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development)
    # You could be stricter: ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# --- REST API Endpoint for React Frontend ---
@app.post("/ask", response_model=AskResponse)
async def http_ask_query(request: AskRequest) -> AskResponse:
    """Synchronous REST endpoint for React app to ask questions.

    This endpoint provides direct access to the RAG pipeline without
    using Inngest, making it suitable for real-time user interactions.

    Args:
        request: AskRequest containing the question and optional interaction_id

    Returns:
        AskResponse with answer, sources, and interaction_id

    Raises:
        HTTPException: If question is empty or processing fails
    """
    logger.info("Received question for /ask: %s", request.question[:100])

    question = request.question
    if not question or not question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty",
        )

    # Generate unique ID for tracking this interaction
    interaction_id = request.interaction_id or str(uuid.uuid4())

    inputs = {
        "question": question,
        "is_kb_relevant": False,
    }

    try:
        # Invoke the compiled graph
        final_state = await app_graph.ainvoke(inputs)

        # Extract response
        generation = final_state.get(
            "generation",
            "I was unable to process your question. Please try again.",
        )
        sources = final_state.get("documents", [])

        logger.info(
            "Successfully processed /ask request (ID: %s)",
            interaction_id,
        )

        return AskResponse(
            answer=generation,
            sources=sources,
            interaction_id=interaction_id,
        )

    except ValidationError as ve:
        # Input guard failed
        logger.warning("Input validation failed for /ask: %s", ve)
        return AskResponse(
            answer=(
                "I'm sorry, I am a mathematical assistant. "
                "I can only help with questions related to math."
            ),
            sources=[],
            interaction_id=interaction_id,
        )

    except Exception as e:
        logger.exception("Error in /ask endpoint: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your question: {str(e)}",
        )


# --- HITL Feedback Endpoint for DSPy Optimization ---
@app.post("/feedback", response_model=FeedbackResponse)
async def http_receive_feedback(feedback: FeedbackRequest) -> FeedbackResponse:
    """Receive and store user feedback for Human-in-the-Loop (HITL) learning.

    This endpoint collects feedback data that can be used to:
    1. Fine-tune the LLM
    2. Optimize prompts with DSPy
    3. Improve retrieval quality
    4. Track system performance over time

    The feedback is stored in JSONL format, where each line is a complete
    JSON object representing one feedback instance. This format is ideal
    for DSPy training and evaluation.

    Args:
        feedback: FeedbackRequest containing interaction details and user feedback

    Returns:
        FeedbackResponse confirming receipt of feedback

    Raises:
        HTTPException: If feedback cannot be saved
    """
    logger.info(
        "Received %s feedback for interaction %s",
        "positive" if feedback.is_good else "negative",
        feedback.interaction_id,
    )

    try:
        # Ensure feedback directory exists
        feedback_path = Path(FEEDBACK_FILE)
        feedback_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare feedback record with metadata for DSPy
        feedback_record = {
            "interaction_id": feedback.interaction_id,
            "question": feedback.question,
            "answer": feedback.answer,
            "is_good": feedback.is_good,
            "corrected_answer": feedback.corrected_answer,
            "sources": feedback.sources or [],
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        # Append to JSONL file (each line is a complete JSON object)
        with open(feedback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_record, ensure_ascii=False) + "\n")

        # Log summary for monitoring
        if feedback.is_good:
            logger.info("✅ Positive feedback logged (ID: %s)", feedback.interaction_id)
        else:
            logger.warning(
                "❌ Negative feedback logged (ID: %s)%s",
                feedback.interaction_id,
                f" with correction: '{feedback.corrected_answer[:50]}...'"
                if feedback.corrected_answer
                else "",
            )

        return FeedbackResponse(
            status="success",
            message="Feedback received and stored successfully",
        )

    except IOError as e:
        logger.error("Failed to write feedback to file: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Failed to save feedback due to file system error",
        )
    except Exception as e:
        logger.exception("Unexpected error while processing feedback: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process feedback: {str(e)}",
        )


# Serve both Inngest functions and REST endpoints
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])
