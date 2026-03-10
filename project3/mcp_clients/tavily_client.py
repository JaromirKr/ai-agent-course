import asyncio
import os

from langchain_mcp_adapters.client import MultiServerMCPClient

async def load_tavily_tools():
    api_key = os.getenv("TAVILY_API_KEY")

    client = MultiServerMCPClient(
        {
            "tavily": {
                "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={api_key}",
                "transport": "streamable_http",
            }
        }
    )

    print("Running Tavily tools...")
    tools = await client.get_tools()
    return tools