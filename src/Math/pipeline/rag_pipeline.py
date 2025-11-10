"""Complete RAG Pipeline with Inngest and Firecrawl MCP for web search.

This module implements a full-featured RAG system with:
- PDF ingestion and vectorization
- Semantic search with relevance scoring
- Automatic web search fallback via Firecrawl MCP
- Input/output validation with Guardrails
- Conversation history management
- Human-in-the-loop feedback collection
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import HTTPException
from guardrails.errors import ValidationError
from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
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

load_dotenv()


class RAGPipeline:
    """Complete RAG Pipeline with Firecrawl MCP for web search capabilities."""

    # Configuration constants
    FEEDBACK_FILE = "feedback.jsonl"
    RELEVANCE_THRESHOLD = 0.70  # Minimum similarity score for KB relevance
    SUMMARY_THRESHOLD = 102400  # Token threshold for history summarization
    MCP_CONFIG_FILE = "mcp_config.json"  # MCP server configuration

    def __init__(self, config: ConfigurationManager | None = None):
        """Initialize RAG Pipeline with all components.

        Args:
            config: Configuration manager instance. If None, creates new instance.
        """
        # Initialize configuration
        self.config_manager = config or ConfigurationManager()
        self.data_ingestion_config = self.config_manager.get_data_ingestion_config()
        self.qdrant_config = self.config_manager.get_data_storing_params()

        # Initialize data components
        self.data_loader = DataLoader(config=self.data_ingestion_config)
        self.qdrant_storage = QdrantStorage(config=self.qdrant_config)

        # Initialize guardrails
        self.input_guard = InputGuard()
        self.output_guard = OutputGuard()

        # LLM configuration
        self.model_name = self.config_manager.config.models[0].parameters.model
        self.base_url = self.config_manager.config.models[0].parameters.base_url
        self.api_key = os.getenv("OPENROUTER_API_KEY")

        # Validate MCP configuration
        self._validate_mcp_config()

        # Initialize Inngest client
        self.inngest_client = inngest.Inngest(
            app_id="rag_app",
            logger=logger,
            is_production=False,
            serializer=inngest.PydanticSerializer(),
        )

        # Build and compile the workflow
        self.app_graph = self._build_workflow()

        # Register Inngest functions
        self._register_inngest_functions()

        logger.info("RAG Pipeline initialized successfully")

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
                    if "mcpServers" not in config:
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

            logger.info("Loaded %d chunks from PDF", len(chunks))
            return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

        def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
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
        inputs = {
            "question": question,
            "is_kb_relevant": False,
            "is_valid": False,
            "documents": [],
            "generation": "",
            "history": [],
            "summary": "",
            "history_tokens": 0,
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

    def _build_workflow(self) -> Any:
        """Build and compile the complete LangGraph workflow.

        The workflow follows this structure:
        1. Validate question (is it math-related?)
        2. Retrieve from knowledge base
        3. Check KB relevance
        4. If relevant: prepare context → generate answer
        5. If not relevant: web search → generate answer
        6. Validate output

        Returns:
            Compiled LangGraph workflow
        """

        def validate_question(state: GraphState) -> dict[str, Any]:
            """Validate that the question is math-related using Guardrails.

            Returns:
                Dictionary with is_valid flag and optional error message
            """
            logger.info("Running validate_question node")
            question = state.question

            try:
                self.input_guard.validate(text_to_validate=question)
                logger.info("✅ Input validation passed")
                return {"is_valid": True}
            except ValidationError as e:
                logger.warning(f"❌ Input validation failed: {e}")
                return {
                    "is_valid": False,
                    "generation": "I can only answer math-related questions. Please ask about mathematical concepts, problems, or theories.",
                }

        def retrieve(state: GraphState) -> dict[str, Any]:
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
                is_kb_relevant = bool(
                    documents and any(doc.strip() for doc in documents)
                )
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

        def should_web_search(state: GraphState) -> str:
            """Routing function: decide whether to use web search or KB generation.

            Returns:
                Next node name: "prepare_context" or "web_search"
            """
            is_kb_relevant = state.is_kb_relevant
            logger.info(f"Routing decision: KB relevant = {is_kb_relevant}")

            if is_kb_relevant:
                logger.info("→ Routing to prepare_context (using KB)")
                return "prepare_context"
            else:
                logger.info("→ Routing to web_search (KB insufficient)")
                return "web_search"

        def prepare_context(state: GraphState) -> dict[str, Any]:
            """Prepare conversation context from history.

            Handles two cases:
            1. History below threshold: Use full history
            2. History above threshold: Summarize with LLM

            Returns:
                Dictionary with summary and updated history_tokens
            """
            logger.debug("Running prepare_context node")
            history = state.history
            question = state.question
            current_history_tokens = state.history_tokens

            if not history:
                logger.debug("No conversation history")
                return {"summary": "", "history_tokens": 0}

            logger.info(
                f"History token count: {current_history_tokens}/{self.SUMMARY_THRESHOLD}"
            )

            if current_history_tokens <= self.SUMMARY_THRESHOLD:
                # Use full history
                logger.info("Using full conversation history")
                prompt_history = []
                for msg in history:
                    role = "User" if msg.get("sender") == "user" else "Assistant"
                    prompt_history.append(f"{role}: {msg.get('text', '')}")
                history_str = "\n".join(prompt_history)

                return {
                    "summary": history_str,
                    "history_tokens": current_history_tokens,
                }
            else:
                # Summarize history
                logger.info("History exceeds threshold. Summarizing...")
                prompt_history = []
                for msg in history:
                    role = "User" if msg.get("sender") == "user" else "Assistant"
                    prompt_history.append(f"{role}: {msg.get('text', '')}")
                history_str = "\n".join(prompt_history)

                summary_prompt = (
                    "You are a helpful summarization assistant. "
                    "Condense the following conversation into a concise summary. "
                    "Preserve key mathematical concepts, formulas, and important details. "
                    "Focus on information relevant to the user's new question.\n\n"
                    "--- CONVERSATION HISTORY ---\n"
                    f"{history_str}\n\n"
                    "--- USER'S NEW QUESTION ---\n"
                    f"{question}\n\n"
                    "Provide a brief, focused summary:\n"
                )

                try:
                    summarizer_llm = ChatOpenAI(
                        model=self.model_name,
                        base_url=self.base_url,
                        api_key=self.api_key,
                    )
                    response = summarizer_llm.invoke(summary_prompt)

                    new_summary = getattr(response, "content", str(response))
                    new_token_count = response.response_metadata.get(
                        "token_usage", {}
                    ).get("total_tokens", 0)

                    logger.info(
                        f"✅ Summarization complete. New tokens: {new_token_count}"
                    )
                    return {"summary": new_summary, "history_tokens": new_token_count}

                except Exception as e:
                    logger.error(f"❌ Summarization failed: {e}")
                    return {"summary": "", "history_tokens": 0}

        async def web_search(state: GraphState) -> dict[str, Any]:
            """Structured Firecrawl MCP web search with schema auto-patching and fallback."""

            logger.info("🌐 Running structured Firecrawl web search (MCP + fallback)")
            question = state.question

            if not Path(self.MCP_CONFIG_FILE).exists():
                logger.error(f"MCP config not found at {self.MCP_CONFIG_FILE}")
                return {
                    "generation": "Web search not configured.",
                    "documents": ["Missing mcp_config.json"],
                }

            try:
                # --- Load MCP configuration ---
                with open(self.MCP_CONFIG_FILE) as f:
                    mcp_servers = json.load(f)

                mcp_client = MultiServerMCPClient(mcp_servers)
                logger.info("✅ MCP client initialized")

                # --- 2️⃣ Load and locate Firecrawl search tool ---
                tools = await mcp_client.get_tools()
                logger.info(f"Available tools: {[t.name for t in tools]}")

                search_tool = next(
                    (t for t in tools if t.name == "firecrawl_search"), None
                )
                if not search_tool:
                    logger.error("❌ No 'firecrawl_search' tool found in MCP registry")
                    return {
                        "generation": "Firecrawl search unavailable.",
                        "documents": [],
                    }

                # --- 3️⃣ Build valid Firecrawl input payload ---
                payload = {
                    "query": question,
                    "sources": [{"type": "web"}],
                    "limit": 5,
                }
                logger.info(f"🔍 Firecrawl payload: {payload}")

                # --- 4️⃣ Call MCP tool directly ---
                try:
                    results = await search_tool.ainvoke(payload)
                    logger.info("✅ Firecrawl MCP search completed")
                except Exception as inner_err:
                    logger.error(
                        f"🔥 Firecrawl MCP call failed: {inner_err}", exc_info=True
                    )
                    return {
                        "generation": "Error during Firecrawl search call.",
                        "documents": [f"Firecrawl error: {str(inner_err)}"],
                    }

                # --- 5️⃣ Summarize results with the LLM ---
                summary_prompt = (
                    "You are a math tutor. Summarize the following web search results clearly:\n\n"
                    f"{results}\n\n"
                    "Focus on key ideas, reasoning, and examples related to the query."
                )

                llm = ChatOpenAI(
                    model=self.model_name,
                    base_url=self.base_url,
                    api_key=self.api_key,
                )

                logger.info("🧠 Summarizing Firecrawl search results with LLM...")
                response = await llm.ainvoke(summary_prompt)
                answer = getattr(response, "content", str(response))

                # --- 6️⃣ Return clean structured result ---
                logger.info("✅ Firecrawl web search succeeded and summarized")
                return {"generation": answer, "documents": [results]}

            except Exception as e:
                logger.error(f"❌ Firecrawl web search failed: {e}", exc_info=True)
                return {
                    "generation": "I encountered an error during Firecrawl web search.",
                    "documents": [f"Firecrawl error: {str(e)}"],
                }

        async def generate(state: GraphState) -> dict[str, Any]:
            """Generate answer using LLM with context and output validation.

            Returns:
                Dictionary with generation and updated history_tokens
            """
            logger.debug("Running generate node")
            question = state.question
            documents = state.documents
            summary = state.summary
            current_history_tokens = state.history_tokens

            # Prepare context from documents
            valid_docs = [doc for doc in documents if doc and doc.strip()]
            context_str = "\n\n".join(valid_docs) if valid_docs else ""

            # Build prompt based on available context
            if context_str:
                prompt = (
                    "You are a helpful math assistant. Use the following context to "
                    "answer the question accurately and clearly.\n\n"
                    "If the context doesn't contain enough information, say so honestly.\n\n"
                    f"--- CONTEXT ---\n{context_str}\n\n"
                )
                if summary:
                    prompt += f"--- CONVERSATION HISTORY ---\n{summary}\n\n"
                prompt += f"--- QUESTION ---\n{question}\n\n--- ANSWER ---\n"
            else:
                prompt = (
                    "You are a helpful math assistant. Answer the following question "
                    "to the best of your ability.\n\n"
                )
                if summary:
                    prompt += f"--- CONVERSATION HISTORY ---\n{summary}\n\n"
                prompt += f"--- QUESTION ---\n{question}\n\n--- ANSWER ---\n"

            try:
                # Generate answer
                llm = ChatOpenAI(
                    model=self.model_name, base_url=self.base_url, api_key=self.api_key
                )
                response = await llm.ainvoke(prompt)

                # Track token usage
                token_usage = response.response_metadata.get("token_usage", {})
                generation_cost = token_usage.get("total_tokens", 0)
                new_total_tokens = current_history_tokens + generation_cost

                content = getattr(response, "content", str(response))

                # Validate output with guardrails
                try:
                    self.output_guard.validate(text_to_validate=content)
                    logger.info(
                        f"✅ Generation complete and validated ({len(content)} chars, {generation_cost} tokens)"
                    )
                except ValidationError as ve:
                    logger.warning(f"⚠️ Output validation failed: {ve}")
                    # Continue anyway but log the issue

                return {"generation": content, "history_tokens": new_total_tokens}

            except ValidationError as ve:
                logger.error(f"❌ Output guardrail rejection: {ve}")
                return {
                    "generation": (
                        "I apologize, but I couldn't generate a satisfactory answer. "
                        "Could you please rephrase your question?"
                    ),
                    "history_tokens": current_history_tokens,
                }

            except Exception as exc:
                logger.exception(f"❌ Generation failed: {exc}")
                return {
                    "generation": (
                        "An error occurred while generating the answer. "
                        "Please try again."
                    ),
                    "history_tokens": current_history_tokens,
                }

        # Build the workflow graph
        logger.info("Building LangGraph workflow...")
        workflow = StateGraph(GraphState)

        # Add all nodes
        workflow.add_node("validate_question", validate_question)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("prepare_context", prepare_context)
        workflow.add_node("web_search", web_search)
        workflow.add_node("generate", generate)

        # Set entry point
        workflow.set_entry_point("validate_question")

        # Define routing functions
        def after_validation(state: GraphState) -> str:
            """Route after validation: continue or end."""
            return "retrieve" if state.is_valid else END

        # Add edges
        workflow.add_conditional_edges(
            "validate_question",
            after_validation,
            {END: END, "retrieve": "retrieve"},
        )

        workflow.add_conditional_edges(
            "retrieve",
            should_web_search,
            {"web_search": "web_search", "prepare_context": "prepare_context"},
        )

        workflow.add_edge("web_search", END)
        workflow.add_edge("prepare_context", "generate")
        workflow.add_edge("generate", END)

        logger.info(" Workflow compiled successfully")
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
        inputs = {
            "question": question,
            "is_kb_relevant": False,
            "is_valid": False,
            "documents": [],
            "generation": "",
            "history": history or [],
            "summary": "",
            "history_tokens": 0,
        }

        try:
            # Execute the workflow
            final_state = await self.app_graph.ainvoke(inputs)

        except Exception as e:
            logger.error(f" Pipeline execution failed: {e}", exc_info=True)
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
            f" Query complete. Answer length: {len(generation)}, Sources: {len(sources)}"
        )

        return {"answer": generation, "sources": sources}

    def get_inngest_client(self) -> inngest.Inngest:
        """Get the Inngest client for FastAPI integration.

        Returns:
            Configured Inngest client instance
        """
        return self.inngest_client
