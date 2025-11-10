from dataclasses import dataclass, field
from pathlib import Path
from pydantic import BaseModel
# from typing import List  # Or from typing import List


@dataclass
class DataIngestion:
    docs: Path
    EMBED_MODEL: str


@dataclass
class DataStoring:
    dimension: int
    collection_name: str
    contexts: list[str]
    sources: set


@dataclass
class RAGChunkAndSrc:
    chunks: list[str]
    source_id: str


class RAGUpsertResult(BaseModel):  # <-- Change inheritance
    ingested: int


@dataclass
class GraphState:
    question: str
    is_kb_relevant: bool
    is_valid: bool = False
    documents: list[str] = field(default_factory=list)
    generation: str = ""
    history: list[str] = field(default_factory=list)
    summary: str = ""
    history_tokens: int = 0
    is_web_search_result: bool = False  # Flag for self-improving RAG
    # Provide a 'default_factory' for mutable types like lists


@dataclass
class AskRequest:
    """Request model for the /ask endpoint."""

    question: str
    interaction_id: str | None = None
    history: list[dict] | None = None


@dataclass
class AskResponse:
    """Response model for the /ask endpoint."""

    answer: str
    sources: list[str]
    interaction_id: str


@dataclass
class FeedbackRequest:
    """Request model for the /feedback endpoint (HITL)."""

    interaction_id: str
    question: str
    answer: str
    is_good: bool
    corrected_answer: str | None = None  # User's suggested correction
    sources: list[str] | None = None  # Context sources for DSPy training


@dataclass
class FeedbackResponse:
    """Response model for the /feedback endpoint."""

    status: str
    message: str
