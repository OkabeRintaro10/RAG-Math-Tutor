"""FastAPI server for RAG pipeline with Inngest integration."""

# Internal Lib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

# External Lib
import inngest.fast_api
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from guardrails.errors import ValidationError

# Local Lib
from src.Math import logger
from src.Math.entity.config_entity import (
    AskRequest,
    AskResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from src.Math.pipeline.rag_pipeline import RAGPipeline

# Initialize RAG Pipeline (single instance for the entire app)
rag_pipeline = RAGPipeline()

# Get Inngest client from pipeline
inngest_client = rag_pipeline.get_inngest_client()

# Initialize FastAPI app
app = FastAPI(title="RAG Inngest Server")

# Add CORS middleware
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

    try:
        # Use the pipeline's query method
        result = await rag_pipeline.query(question)

        # Extract response
        generation = result.get(
            "answer",
            "I was unable to process your question. Please try again.",
        )
        sources = result.get("sources", [])

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
        feedback_path = Path(rag_pipeline.FEEDBACK_FILE)
        feedback_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare feedback record with metadata for DSPy
        feedback_record = {
            "interaction_id": feedback.interaction_id,
            "question": feedback.question,
            "answer": feedback.answer,
            "is_good": feedback.is_good,
            "corrected_answer": feedback.corrected_answer,
            "sources": feedback.sources or [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
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


# --- Health Check Endpoint ---
@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint to verify server is running.

    Returns:
        Dictionary with status message
    """
    return {"status": "healthy", "service": "RAG Inngest Server"}


# --- Inngest PDF Ingestion Endpoint (Trigger via HTTP) ---
@app.post("/ingest")
async def trigger_pdf_ingestion(pdf_path: str, source_id: str | None = None):
    """Trigger PDF ingestion via Inngest event.

    Args:
        pdf_path: Path to the PDF file to ingest
        source_id: Optional unique identifier for the source

    Returns:
        Dictionary with ingestion trigger status

    Raises:
        HTTPException: If PDF path is invalid or ingestion fails
    """
    if not pdf_path or not pdf_path.strip():
        raise HTTPException(
            status_code=400,
            detail="PDF path cannot be empty",
        )

    try:
        # Send event to Inngest to trigger ingestion
        await inngest_client.send(
            inngest.Event(
                name="rag/ingest_pdf",
                data={
                    "pdf_path": pdf_path,
                    "source_id": source_id or pdf_path,
                },
            )
        )

        logger.info("Triggered PDF ingestion for: %s", pdf_path)
        return {
            "status": "triggered",
            "message": f"PDF ingestion started for {pdf_path}",
        }

    except Exception as e:
        logger.exception("Failed to trigger PDF ingestion: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to trigger ingestion: {str(e)}",
        )


# Serve Inngest functions alongside REST endpoints
inngest.fast_api.serve(
    app,
    inngest_client,
    # Pass the bound methods directly - they're already registered
    [rag_pipeline._rag_ingest_pdf, rag_pipeline._rag_query_pdf_ai],
)
