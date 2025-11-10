# %%
import os

print(os.getcwd())
os.chdir("../")
print(os.getcwd())


# %%
from dotenv import load_dotenv

load_dotenv()

# %%
from src.Math.config.configuration import ConfigurationManager


# %%
config_manager = ConfigurationManager()
model_name = config_manager.config.models[0].parameters.model
base_url = config_manager.config.models[0].parameters.base_url
api_key = os.getenv("OPENROUTER_API_KEY")

# %%
import asyncio
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient


async def main():
    # Configure MCP server
    config = {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            }
        }
    }

    client = MCPClient.from_dict(config)
    llm = ChatOpenAI(model=model_name, base_url=base_url, api_key=api_key)
    agent = MCPAgent(llm=llm, client=client)

    result = await agent.run("List all files in the directory")
    print(result)


asyncio.run(main())
