"""
A custom, pure-Python MCP server using the official 'firecrawl-py' SDK.
This now runs as an HTTP server to avoid stdio/anyio conflicts.
"""

import os
import sys
import json
from mcp.server.fastmcp import FastMCP
from firecrawl import FirecrawlApp
from dotenv import load_dotenv

# --- 1. Load Config & API Key ---
load_dotenv()
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

if not FIRECRAWL_API_KEY:
    print(
        "Error: FIRECRAWL_API_KEY not found in .env file. MCP server cannot start.",
        file=sys.stderr,
    )
    sys.exit(1)

# --- 2. Create the Server ---
mcp = FastMCP(
    name="py-firecrawl-mcp",
)

try:
    firecrawl_app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
    print("--- FirecrawlApp initialized successfully ---", file=sys.stderr)
except Exception as e:
    print(f"Error initializing FirecrawlApp: {e}", file=sys.stderr)
    sys.exit(1)


# --- 3. Define the Search Tool ---
@mcp.tool()
def firecrawl_search(query: str, num_results: int = 5) -> str:
    """
    Performs a web search using the Firecrawl API.
    Returns results as a JSON string.
    """
    print(f"MCP Server: Received search query: {query}", file=sys.stderr)
    try:
        results = firecrawl_app.search(query=query, page_options={"limit": num_results})
        print(f"MCP Server: Found {len(results)} results.", file=sys.stderr)
        return json.dumps({"data": results, "error": None})
    except Exception as e:
        print(f"MCP Server: Error calling Firecrawl search: {e}", file=sys.stderr)
        return json.dumps({"data": [], "error": str(e)})


# --- 4. Define the Scrape Tool ---
@mcp.tool()
def firecrawl_scrape(url: str) -> str:
    """
    Scrapes a single URL using the Firecrawl API.
    Returns a JSON string of the scraped data.
    """
    print(f"MCP Server: Received scrape request for: {url}", file=sys.stderr)
    try:
        scraped_data = firecrawl_app.scrape_url(url=url)
        print(
            f"MCP Server: Scraped {len(scraped_data.get('markdown', ''))} chars from {url}",
            file=sys.stderr,
        )
        return json.dumps(scraped_data)
    except Exception as e:
        print(f"MCP Server: Error scraping {url}: {e}", file=sys.stderr)
        return json.dumps({"error": str(e), "markdown": None, "content": None})


# --- 5. Run the Server ---
if __name__ == "__main__":
    # --- CHANGED ---
    # We now run as an HTTP server on 127.0.0.1:8001
    # This must be run in its own terminal.
    print(
        "--- Starting custom Python (Firecrawl) MCP HTTP server on 127.0.0.1:8001 ---",
        file=sys.stderr,
    )
    mcp.run(transport="http", port=8001)
