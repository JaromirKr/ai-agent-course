import asyncio

from dotenv import load_dotenv

from mcp_clients.tavily_client import load_tavily_tools
from tools.extract_keywords_tfidf import extract_keywords_tfidf
from agents.researcher import create_researcher
from agents.analyst import create_analyst
from graph.workflow import build_workflow

load_dotenv()

async def main():
    """
    you need to add OPENAI_API_KEY, TAVILY_API_KEY, btw for tf idf it is better not to limit it to max one sentence,
    but he was terribly chatty and the answer took a long time to come, hence the limitation in the prompt for testing
    """

    # MCP tooly (Tavily)
    tavily_tools = await load_tavily_tools()

    # Custom TF-IDF tool
    custom_tools = [extract_keywords_tfidf]

    # Agents
    researcher = create_researcher(tavily_tools, custom_tools)
    analyst = create_analyst()

    # Workflow
    app = build_workflow(researcher, analyst)

    # The input query
    result = await app.ainvoke({"question": "V max jedné větě jaký je předpokládaný vývoj ceny zlata a stříbra?"})

    print("\n=== Analysis ===\n")
    print(result["analysis"])

if __name__ == "__main__":
    asyncio.run(main())
