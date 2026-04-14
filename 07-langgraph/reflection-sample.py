
from langgraph.graph import MessageGraph, END
from langchain_core.messages import HumanMessage, AIMessage

# Node that generates an initial response
def generate_answer(state):
    # (In practice, call an LLM here)
    answer = "This is my first attempt."
    return {"messages": state["messages"] + [AIMessage(content=answer)]}

# Node that critiques and refines the previous answer
def critique_answer(state):
    # (In practice, call LLM to critique)
    critique = "The answer is incomplete; add more detail."
    return {"messages": state["messages"] + [AIMessage(content=critique)]}

builder = MessageGraph()
builder.add_node("generate", generate_answer)
builder.add_node("reflect", critique_answer)
builder.set_entry_point("generate")

# Loop control: alternate until max iterations
MAX_STEPS = 3
def should_continue(state):
    return "reflect" if len(state["messages"]) < 2*MAX_STEPS else END

builder.add_conditional_edges("generate", should_continue)
builder.add_edge("reflect", "generate")
graph = builder.compile()

# Run the reflection agent
initial_message = HumanMessage(content="Explain photosynthesis.")
result = graph.invoke({"messages": [initial_message]})
print(result["messages"][-1])  # Final answer or critique