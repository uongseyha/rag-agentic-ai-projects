from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

# Simple state with messages and a step counter
def call_model(state):
    # (LLM reasons; may request an action or give an answer)
    last = state["messages"][-1]
    if "weather" in last:
        # chain-of-thought leading to an action
        thought = AIMessage(content="Let me find the weather for you.")
        return {"messages": state["messages"] + [thought]}
    else:
        # final answer
        answer = AIMessage(content="It's sunny in NYC today.")
        return {"messages": state["messages"] + [answer]}

def call_tool(state):
    # (Simulate a weather API/tool result)
    tool_result = AIMessage(content="Weather(temperature=75F, condition=sunny)")
    return {"messages": state["messages"] + [tool_result]}

# Decide whether to act or finish based on last message
def next_step(state):
    last = state["messages"][-1]
    if "find the weather" in last:
        return "tools"
    return "end"

graph = StateGraph(dict)  # using a plain dict state
graph.add_node("think", call_model)
graph.add_node("act", call_tool)
graph.set_entry_point("think")

# If the model’s message triggers an action, go to 'act'; else end.
graph.add_conditional_edges("think", next_step, {"tools": "act", "end": END})
graph.add_edge("act", "think")
compiled = graph.compile()
result = compiled.invoke({"messages": [HumanMessage(content="What is the weather in NYC?")]})
print(result["messages"][-1])  # Final assistant answer
