from typing import Any
import json
import asyncio
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

from src.Math import logger
from src.Math.config.configuration import ConfigurationManager
from src.Math.entity.config_entity import GraphState


class WebSearch:
    def __init__(self, config_file: str):
        self.MCP_CONFIG_FILE = config_file

    async def web_search(self, state: GraphState) -> dict[str, Any]:
        """Perform a web search using Firecrawl MCP and return the raw results.

        This node retrieves context from the web but does not generate a final answer.
        The results are placed in the 'documents' field of the state for the 'generate'
        node to use.

        Returns:
            A dictionary with the 'documents' key updated with web search results.
        """
        logger.info("🌐 Running Firecrawl web search node...")
        question = state.question

        if not Path(self.MCP_CONFIG_FILE).exists():
            logger.error(f"MCP config not found at {self.MCP_CONFIG_FILE}")
            return {
                "documents": [
                    "Web search is not configured (mcp_config.json is missing)."
                ]
            }

        try:
            # Load MCP configuration
            with open(self.MCP_CONFIG_FILE) as f:
                mcp_servers = json.load(f)

            mcp_client = MultiServerMCPClient(mcp_servers)

            # Locate the Firecrawl search tool
            tools = await mcp_client.get_tools()
            search_tool = next((t for t in tools if t.name == "firecrawl_search"), None)

            if not search_tool:
                logger.error("❌ No 'firecrawl_search' tool found in MCP registry")
                return {
                    "documents": [
                        "Firecrawl search tool is not available in the MCP registry."
                    ]
                }

            # Build the payload and invoke the tool
            payload = {"query": question, "sources": [{"type": "web"}], "limit": 5}
            logger.info(f"🔍 Calling Firecrawl with payload: {payload}")
            results = await search_tool.ainvoke(payload)
            logger.info("✅ Firecrawl MCP search completed successfully.")

            # Format results into a single string for the generate node
            formatted_results = (
                f"Web search results for the query '{question}':\n\n"
                + json.dumps(results, indent=2)
            )

            return {"documents": [formatted_results], "is_web_search_result": True}

        except asyncio.CancelledError:
            logger.error("❌ Firecrawl web search was cancelled.", exc_info=True)
            return {"documents": ["The web search was cancelled."]}

        except Exception as e:
            logger.error(f"❌ Firecrawl web search failed: {e}", exc_info=True)
            return {"documents": [f"An error occurred during web search: {str(e)}"]}
