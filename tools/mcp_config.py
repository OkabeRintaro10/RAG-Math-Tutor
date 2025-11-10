"""
Simplified MCP configuration.
Provides constants for the RAG pipeline and a validation function.
Now configured for the Firecrawl-py SDK.
"""

from __future__ import annotations
import os
from src.Math import logger
from dotenv import load_dotenv

load_dotenv()


class MCPConfig:
    """Configuration for our custom MCP client/server setup."""

    # --- We now check for the Firecrawl key ---
    FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

    # --- KEPT FOR COMPATIBILITY ---
    # These constants may be used by test.py or rag_pipeline.py
    MAX_SEARCH_RESULTS = 3
    SCRAPE_FORMATS = ["markdown"]
    CRAWL_MAX_DEPTH = 1

    @classmethod
    def validate_config(cls) -> bool:
        """Validate that the Firecrawl API key is present."""
        if not cls.FIRECRAWL_API_KEY:
            logger.error(
                "FIRECRAWL_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )
            raise ValueError("FIRECRAWL_API_KEY not found.")

        logger.info("FIRECRAWL_API_KEY (MCP) validated successfully.")
        return True
