from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

def create_analyst():
    llm = ChatOpenAI(model="gpt-5-nano")

    prompt = """
You are the Analyst.
You receive the output from the Researcher in the following structure:
Code
RAW_TEXT:
...
KEYWORDS:
...
Your tasks:
    identify key trends
    highlight risks and opportunities
    produce a concise final analysis
    propose 1–2 actionable recommendations
Rules:
    You do not use any tools.
    You do not perform additional research.
    Base your analysis only on the Researcher’s output.
"""

    agent = create_agent(
        model=llm,
        tools=[],
        system_prompt=prompt,
    )

    return agent
