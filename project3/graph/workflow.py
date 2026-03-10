from typing import TypedDict

from langgraph.graph import StateGraph, END

from project3.graph.visualizer import visualize


class State(TypedDict):
    question: str
    research: str
    analysis: str

def build_workflow(researcher, analyst):
    async def researcher_node(state: State):
        print("Research node running...")

        result = await researcher.ainvoke(
            { "messages": [{"role": "user", "content": f"{state['question']}"}] }
        )

        state["research"] = get_last_ai_message(result["messages"])
        return state

    async def analyst_node(state: State):
        print("Analysis node running...")
        result = await analyst.ainvoke(
            { "messages": [{"role": "user", "content": f"{state['research']}"}] }
        )
        state["analysis"] = get_last_ai_message(result["messages"])
        return state

    def get_last_ai_message(messages):
        last_ai = next(
            (m for m in reversed(messages) if m.type == "ai"),
            None
        )
        return last_ai.content

    graph = StateGraph(State)
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)

    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", END)
    graph = graph.compile()

    # visualize(graph, "graph.png")

    return graph
