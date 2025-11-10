"""
MCP Client for our *custom* Python server (tools/mcp_server.py).
This client now connects via HTTP to avoid stdio/anyio conflicts.
"""

from __future__ import annotations
import asyncio
import json
from typing import Any

from mcp import ClientSession
from mcp.client.http import http_client  # <-- CHANGED
from src.Math import logger
from tools.mcp_config import MCPConfig


class FirecrawlMCPClient:
    """
    Client for interacting with our custom self-hosted Python MCP server.
    Connects via HTTP.
    """

    SERVER_URL = "http://127.0.0.1:8001"

    def __init__(self):
        """Initialize MCP client to connect to our HTTP server."""
        self.session: ClientSession | None = None
        self._connected = False
        self.http_cm: Any = None  # To store the http context manager
        self.read: Any = None
        self.write: Any = None

    async def connect(self):
        """Connect to the MCP HTTP server."""
        if self._connected:
            return

        try:
            logger.info(
                f"Connecting to custom Python MCP HTTP server at {self.SERVER_URL}..."
            )

            # --- CHANGED ---
            # Use http_client instead of stdio_client
            self.http_cm = http_client(url=self.SERVER_URL)
            self.read, self.write = await self.http_cm.__aenter__()
            self.session = ClientSession(self.read, self.write)
            await self.session.__aenter__()

            await self.session.initialize()
            tools = await self.session.list_tools()
            logger.info(
                f"Connected to custom MCP. Available tools: {[t.name for t in tools.tools]}"
            )
            self._connected = True

        except Exception as e:
            logger.error(f"Failed to connect to custom MCP server: {e}")
            if self.session:
                await self.session.__aexit__(type(e), e, e.__traceback__)
            if self.http_cm:
                await self.http_cm.__aexit__(type(e), e, e.__traceback__)
            self._connected = False
            raise

    async def disconnect(self):
        """Disconnect from the MCP server."""
        if not self._connected or not self.session or not self.http_cm:
            return
        try:
            await self.session.__aexit__(None, None, None)
            await self.http_cm.__aexit__(None, None, None)
            logger.info("Disconnected from custom MCP server")
        except Exception as e:
            logger.error(f"Error disconnecting from MCP server: {e}")
        finally:
            self.session = None
            self.http_cm = None
            self.read = None
            self.write = None
            self._connected = False

    async def search_web(
        self,
        query: str,
        max_results: int = 5,
        **kwargs,  # Added **kwargs for compatibility
    ) -> dict[str, Any]:
        """Search the web using our custom Firecrawl-py MCP server."""
        if not self._connected:
            await self.connect()
        if not self.session:
            raise ConnectionError("Not connected to MCP server.")

        logger.info(f"Searching web with Firecrawl-py: {query}")

        try:
            # Call the 'firecrawl_search' tool from our mcp_server.py
            result = await self.session.call_tool(
                "firecrawl_search",
                arguments={
                    "query": query,
                    "num_results": max_results,
                },
            )

            if not result.content:
                return {"data": [], "error": "No content returned"}

            content_text = "".join(
                item.text for item in result.content if hasattr(item, "text")
            )

            try:
                # The server now returns a dict {"data": [...], "error": ...}
                search_data = json.loads(content_text)
                logger.info(
                    f"Search returned {len(search_data.get('data', []))} results"
                )
                return search_data

            except json.JSONDecodeError:
                logger.warning("Could not parse search results as JSON")
                return {"data": [], "raw": content_text}

        except Exception as e:
            logger.error(f"Firecrawl-py search failed: {e}", exc_info=True)
            return {"data": [], "error": str(e)}

    async def scrape_url(
        self,
        url: str,
        **kwargs,  # Added **kwargs for compatibility
    ) -> dict[str, Any]:
        """Scrape a specific URL using our custom Firecrawl-py MCP tool."""
        if not self._connected:
            await self.connect()
        if not self.session:
            raise ConnectionError("Not connected to MCP server.")

        logger.info(f"Scraping URL with custom Firecrawl-py scraper: {url}")

        try:
            # Call the 'firecrawl_scrape' tool from our mcp_server.py
            result = await self.session.call_tool(
                "firecrawl_scrape",
                arguments={"url": url},
            )

            if not result.content:
                return {"error": "No content returned", "url": url}

            content_text = "".join(
                item.text for item in result.content if hasattr(item, "text")
            )

            try:
                # The server returns the full scraped data dict
                scrape_data = json.loads(content_text)
                logger.info(f"Successfully scraped {url}")
                return scrape_data
            except json.JSONDecodeError:
                return {"content": content_text, "url": url}

        except Exception as e:
            logger.error(f"Custom scrape failed: {e}", exc_info=True)
            return {"error": str(e), "url": url}

    async def crawl_website(
        self,
        **kwargs,  # Accept all args
    ) -> dict[str, Any]:
        """Stub for crawl_website. Firecrawl-py SDK has crawl, but not implemented here."""
        logger.warning("crawl_website is not implemented in this custom server setup.")
        # Note: You could easily add this to mcp_server.py if needed
        return {"data": [], "error": "crawl_website is not implemented"}

    async def search_and_extract(
        self, query: str, max_results: int = 3
    ) -> list[dict[str, Any]]:
        """
        Search and extract clean content from results.
        1. Search with Firecrawl.
        2. Scrape each result URL.
        """
        logger.info(f"Running search_and_extract for: {query}")
        search_results = await self.search_web(query, max_results=max_results)

        if not search_results.get("data"):
            logger.warning("No search results found")
            return []

        # 2. Extract content from each result
        tasks = []
        for result in search_results["data"][:max_results]:
            url = result.get("url")
            if not url:
                continue
            tasks.append(self.scrape_url(url))

        # Run scrape tasks concurrently
        scrape_results = await asyncio.gather(*tasks)

        # Format final results
        extracted_contents = []
        for scraped in scrape_results:
            if scraped.get("markdown"):
                extracted_contents.append(
                    {
                        "url": scraped.get("metadata", {}).get("sourceURL", "N/A"),
                        "title": scraped.get("metadata", {}).get("title", "No Title"),
                        "content": scraped["markdown"],
                        "source": "scrape",
                    }
                )

        logger.info(f"Extracted content from {len(extracted_contents)} sources")
        return extracted_contents
