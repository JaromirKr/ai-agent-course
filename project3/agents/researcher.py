from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

def create_researcher(mcp_tools, custom_tools):
    llm = ChatOpenAI(model="gpt-5-nano")

    prompt = """
You are the Researcher.
Your tasks:
    Use MCP tools (Tavily) to gather information about the topic.
    When calling the Tavily search tool, ALWAYS set topic="general".
    Never use any other topic value.
    Use the TF‑IDF keyword extraction tool to extract key terms from the gathered text.
    Produce a structured output in the following format:

Code
RAW_TEXT:
<your synthesized summary>
KEYWORDS:
<comma‑separated list of extracted keywords>

Additional rules:
    Do not provide final recommendations or conclusions.
    Your job is research and extraction, not interpretation.
"""

    tools = list(mcp_tools) + list(custom_tools)

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt,
    )

    return agent
