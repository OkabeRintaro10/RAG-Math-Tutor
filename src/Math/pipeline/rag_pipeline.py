"""Complete RAG Pipeline with Inngest and Firecrawl MCP for web search.

This module implements a full-featured RAG system with:
- PDF ingestion and vectorization
- Semantic search with relevance scoring
- Automatic web search fallback via Firecrawl MCP
- Input/output validation with Guardrails
- Conversation history management
- Human-in-the-loop feedback collection
"""

from pathlib import Path
import uuid
import json
import datetime
from typing import Any
import os

from dotenv import load_dotenv
from fastapi import HTTPException
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache


import inngest
import inngest.fast_api

from src.Math.components.data_ingestion import DataLoader
from src.Math.components.data_storing import QdrantStorage
from guard.guardrails import InputGuard, OutputGuard
from src.Math.config.configuration import ConfigurationManager
from src.Math.components.data_validate import ValidateQuestion
from src.Math.components.query_rewrite import QueryRewriter
from src.Math.components.prepare_context import Prepare_Context
from src.Math.components.web_search import WebSearch
from src.Math.components.data_retrieve import DataRetrieve
from src.Math.components.answer_generate import Generate
from src.Math.components.data_store_and_update import StoreAndUpdate

from src.Math.entity.config_entity import (
    RAGChunkAndSrc,
    RAGUpsertResult,
    GraphState,
)
from src.Math import logger

load_dotenv()


class RAGPipeline:
    """Complete RAG Pipeline with Firecrawl MCP for web search capabilities."""

    # Configuration constants
    FEEDBACK_FILE = "feedback.jsonl"
    RELEVANCE_THRESHOLD = 0.70  # Minimum similarity score for KB relevance
    SUMMARY_THRESHOLD = 102400  # Token threshold for history summarization
    MCP_CONFIG_FILE = "mcp_config.json"  # MCP server configuration

    def __init__(self, config: ConfigurationManager | None = None) -> None:
        """Initialize RAG Pipeline with all components.

        Args:
            config: Configuration manager instance. If None, creates new instance.
        """
        self._initialize_configuration(config)
        self._initialize_guardrails()
        self._initialize_llm_config()
        self._validate_mcp_config()
        self._initialize_inngest_client()

        self._initialize_graph_components()
        self.app_graph = self._build_workflow()
        self._register_inngest_functions()

        logger.info("RAG Pipeline initialized successfully")

    def _initialize_configuration(self, config: ConfigurationManager | None) -> None:
        """Initialize configuration manager and related configs."""
        self.config_manager = config or ConfigurationManager()
        self.data_loader = DataLoader(self.config_manager.get_data_ingestion_config())
        self.qdrant_storage = QdrantStorage(
            self.config_manager.get_data_storing_params()
        )

    def _initialize_guardrails(self) -> None:
        """Initialize input and output guardrails."""
        self.input_guard = InputGuard()
        self.output_guard = OutputGuard()

    def _initialize_llm_config(self) -> None:
        """Initialize LLM configuration and set global cache."""
        # Set up in-memory cache for all LLM calls
        set_llm_cache(InMemoryCache())
        logger.info("LLM cache enabled (in-memory)")

        llm_parameters = self.config_manager.config.models[0].parameters
        self.model_name = llm_parameters.model
        self.base_url = llm_parameters.base_url
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            openai_api_key=self.api_key,
            base_url=self.base_url,
        )

    def _initialize_graph_components(self) -> None:
        """Initialize data loading and storage components."""
        # Initialize components
        self.validate_question = ValidateQuestion(input_guard=self.input_guard)
        self.retrieve = DataRetrieve(
            config=self.config_manager,
            DataLoader=self.data_loader,
            QdrantStorage=self.qdrant_storage,
            RELEVANCE_THRESHOLD=self.RELEVANCE_THRESHOLD,
        )
        self.prepare_context = Prepare_Context(
            llm=self.llm, summary_threshold=self.SUMMARY_THRESHOLD
        )
        self.web_search = WebSearch(config_file=self.MCP_CONFIG_FILE)
        self.generate = Generate(output_guard=self.output_guard, llm=self.llm)
        self.store_and_update = StoreAndUpdate(
            data_loader=self.data_loader,
            qdrant_storage=self.qdrant_storage,
        )
        self.query_rewriter = QueryRewriter(llm=self.llm)

    def _initialize_inngest_client(self) -> None:
        """Initialize the Inngest client."""
        self.inngest_client = inngest.Inngest(
            app_id="rag_app",
            logger=logger,
            is_production=False,
            serializer=inngest.PydanticSerializer(),
        )

    def _validate_mcp_config(self) -> None:
        """Validate MCP configuration file exists and is valid."""
        if not Path(self.MCP_CONFIG_FILE).exists():
            logger.warning(
                f"MCP config file not found at {self.MCP_CONFIG_FILE}. "
                "Web search will be disabled."
            )
            logger.info(
                "Create mcp_config.json with your Firecrawl configuration to enable web search."
            )
        else:
            try:
                with open(self.MCP_CONFIG_FILE) as f:
                    config = json.load(f)
                    if "mcpServers" in config:
                        logger.error("Invalid MCP config: missing 'mcpServers' key")
                    else:
                        logger.info("MCP configuration validated successfully")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in MCP config: {e}")

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
        """Register Inngest functions for async processing."""
        self._rag_ingest_pdf_fn = self.inngest_client.create_function(
            fn_id="RAG: Ingest PDF",
            trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
            throttle=inngest.Throttle(limit=2, period=datetime.timedelta(minutes=1)),
            rate_limit=inngest.RateLimit(
                limit=1,
                period=datetime.timedelta(hours=4),
                key="event.data.source_id",
            ),
        )(self.rag_ingest_pdf_handler)

        self._rag_query_pdf_ai_fn = self.inngest_client.create_function(
            fn_id="RAG: Query PDF",
            trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
        )(self.rag_query_pdf_ai_handler)

        logger.info("Inngest functions registered successfully")

    async def rag_ingest_pdf_handler(self, ctx: inngest.Context) -> dict[str, Any]:
        """Load, chunk, embed and upsert a PDF into Qdrant.

        This handler is triggered by Inngest events and processes PDFs in the background.

        Args:
            ctx: Inngest context containing event data with pdf_path and optional source_id

        Returns:
            Dictionary with ingestion results (number of chunks ingested)

        Raises:
            FileNotFoundError: If the PDF file doesn't exist
        """
        logger.info("Starting RAG ingest PDF function")

        def _load_and_chunk_pdf(ctx_inner: inngest.Context) -> RAGChunkAndSrc:
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

            logger.info("Loaded %d chunks from PDF", len(chunks))
            return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

        def _embed_and_upsert_chunks(
            chunks_and_src: RAGChunkAndSrc,
        ) -> RAGUpsertResult:
            """Embed chunks and upsert to Qdrant."""
            chunks = chunks_and_src.chunks
            source_id = chunks_and_src.source_id

            if not chunks:
                logger.info("No chunks to upsert for source: %s", source_id)
                return RAGUpsertResult(ingested=0)

            logger.info("Embedding %d chunks for source %s", len(chunks), source_id)

            # Create embeddings for all chunks
            vecs = self.data_loader.embed_texts(texts=chunks)

            # Generate stable IDs for each chunk
            ids = self._make_ids(source_id=source_id, count=len(chunks))

            # Create payloads with metadata
            payloads = [
                {"source": source_id, "text": chunks[i]} for i in range(len(chunks))
            ]

            logger.info("Upserting %d vectors to Qdrant", len(vecs))
            self.qdrant_storage.upsert(ids=ids, vectors=vecs, payloads=payloads)

            return RAGUpsertResult(ingested=len(chunks))

        # Execute load and upsert steps
        chunks_and_src = await ctx.step.run(
            "load-and-chunk",
            lambda: _load_and_chunk_pdf(ctx),
            output_type=RAGChunkAndSrc,
        )

        upsert_result = await ctx.step.run(
            "embed-and-upsert",
            lambda: _embed_and_upsert_chunks(chunks_and_src),
            output_type=RAGUpsertResult,
        )

        logger.info("Ingest completed: %d chunks ingested", upsert_result.ingested)
        return upsert_result.model_dump()

    async def rag_query_pdf_ai_handler(self, ctx: inngest.Context) -> dict[str, Any]:
        """Answer questions using RAG pipeline with LangGraph workflow.

        Args:
            ctx: Inngest context containing event data with question

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

        # Initialize state
        inputs: GraphState = {
            "question": question,
            "is_kb_relevant": False,
            "is_valid": False,
            "documents": [],
            "generation": "",
            "history": [],
            "summary": "",
            "history_tokens": 0,
            "is_web_search_result": False,
        }

        try:
            # Execute the graph
            final_state = await self.app_graph.ainvoke(inputs)
        except Exception as e:
            logger.error(f"Graph invocation failed: {e}", exc_info=True)
            return {
                "answer": "I encountered an error processing your question. Please try again.",
                "sources": [],
            }

        # Extract results
        generation = final_state.get(
            "generation",
            "I was unable to process your question. Please try again.",
        )
        sources = final_state.get("documents", [])

        return {"answer": generation, "sources": sources}

    def _should_web_search(self, state: GraphState) -> str:
        """Routing function: decide whether to use web search or generate from KB."""
        is_kb_relevant = state.is_kb_relevant
        logger.info(f"Routing decision: KB relevant = {is_kb_relevant}")

        if is_kb_relevant:
            logger.info("→ Routing to 'prepare_context' (using KB)")
            return "prepare_context"
        else:
            logger.info("→ Routing to 'web_search' (KB insufficient)")
            return "web_search"

    def _build_workflow(self) -> StateGraph:
        """Build and compile the complete, self-improving LangGraph workflow.

        The robust workflow follows this structure:
        1. Validate question.
        2. Retrieve from knowledge base.
        3. Prepare context (for conversation history).
        4. Decide: web search or generate from KB.
        5. (If needed) Perform web search.
        6. Generate the final answer.
        7. (If from web) Store the new Q&A pair back into the knowledge base.
        8. End.

        Returns:
            Compiled LangGraph workflow.
        """
        logger.info("Building self-improving LangGraph workflow...")
        workflow = StateGraph(GraphState)

        # Add all nodes
        workflow.add_node("rewrite_query", self.query_rewriter.rewrite_query)
        workflow.add_node("validate_question", self.validate_question.validate_question)
        workflow.add_node("retrieve", self.retrieve.retrieve)
        workflow.add_node("prepare_context", self.prepare_context.prepare_context)
        workflow.add_node("web_search", self.web_search.web_search)
        workflow.add_node("generate", self.generate.generate)
        workflow.add_node("store_web_search_answer", self.store_and_update.store_answer)

        # --- Define the graph edges ---

        # 1. Entry point
        workflow.set_entry_point("rewrite_query")

        # 2. After validation, retrieve from KB or end
        workflow.add_edge("rewrite_query", "validate_question")

        # 3. After validation, route as normal
        def after_validation(state: GraphState) -> str:
            """Route after validation: continue or end."""
            return "retrieve" if state.is_valid else END

        workflow.add_conditional_edges(
            "validate_question",
            after_validation,
            {END: END, "retrieve": "retrieve"},
        )

        # 3. The rest of the graph logic is the same
        workflow.add_conditional_edges(
            "retrieve",
            self._should_web_search,
            {"web_search": "web_search", "prepare_context": "prepare_context"},
        )

        workflow.add_edge("web_search", "generate")
        workflow.add_edge("prepare_context", "generate")

        # 6. After generation, decide whether to store the result
        def decide_to_store(state: GraphState) -> str:
            if state.is_web_search_result:
                logger.info(
                    "→ Routing to 'store_web_search_answer' to learn from web search."
                )
                return "store_web_search_answer"
            else:
                logger.info("→ Routing to END (KB answer, no storage needed).")
                return END

        workflow.add_conditional_edges(
            "generate",
            decide_to_store,
            {"store_web_search_answer": "store_web_search_answer", END: END},
        )

        # 7. After storing, end the workflow
        workflow.add_edge("store_web_search_answer", END)

        logger.info("✅ Self-improving workflow compiled successfully.")
        return workflow.compile()

    async def query(
        self, question: str, history: list[dict] | None = None
    ) -> dict[str, Any]:
        """Query the RAG pipeline directly (without Inngest).

        This is the main entry point for synchronous question answering.

        Args:
            question: The user's question
            history: Optional conversation history

        Returns:
            Dictionary with 'answer' and 'sources' keys

        Raises:
            ValueError: If question is empty
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        logger.info(f"Processing query: '{question[:50]}...'")

        # Initialize state
        inputs: GraphState = {
            "question": question,
            "is_kb_relevant": False,
            "is_valid": False,
            "documents": [],
            "generation": "",
            "history": history or [],
            "summary": "",
            "history_tokens": 0,
            "is_web_search_result": False,
        }

        try:
            # Execute the workflow
            final_state = await self.app_graph.ainvoke(inputs)

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            return {
                "answer": (
                    "I encountered an error processing your question. "
                    "Please try again or rephrase your question."
                ),
                "sources": [],
            }

        # Extract results
        generation = final_state.get(
            "generation",
            "I was unable to process your question. Please try again.",
        )
        sources = final_state.get("documents", [])

        logger.info(
            f"Query complete. Answer length: {len(generation)}, Sources: {len(sources)}"
        )

        return {"answer": generation, "sources": sources}

    def get_inngest_client(self) -> inngest.Inngest:
        """Get the Inngest client for FastAPI integration.

        Returns:
            Configured Inngest client instance
        """
        return self.inngest_client
