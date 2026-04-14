from langgraph.graph import MessageGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

def draft_answer(state):
    # (LLM draft; could also generate search query)
    response = "The capital of France is Paris."
    return {"messages": state["messages"] + [AIMessage(content=response)]}

def execute_tools(state):
    # (Simulate external info; e.g., search results)
    info = "París (France) - capital: Paris (en.wikipedia.org)"
    return {"messages": state["messages"] + [SystemMessage(content=info)]}

def revise_answer(state):
    # (LLM re-evaluates answer using info)
    revision = "Yes, France’s capital is Paris. I've verified this."
    return {"messages": state["messages"] + [AIMessage(content=revision)]}

builder = MessageGraph()
builder.add_node("draft", draft_answer)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revise_answer)
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")

# Loop control: stop after N iterations
MAX_LOOPS = 2
def continue_reflexion(state):
    # Count assistant messages to determine iteration
    iteration = sum(1 for m in state["messages"] if isinstance(m, AIMessage))
    return "execute_tools" if iteration <= MAX_LOOPS else END

builder.add_conditional_edges("revise", continue_reflexion)
builder.set_entry_point("draft")
graph = builder.compile()
initial_message = HumanMessage(content="What is the capital of France?")
result = graph.invoke({"messages": [initial_message]}) # Final revised answer

#This agent uses a **built-in search or tool** (`execute_tools`) to ground its critique. The revise node then updates the answer explicitly (e.g., adding evidence or fixing errors). The process stops when the agent judges the answer is good or after a set number of loops.
#- **When to use**: Reflexion is ideal when accuracy or factual grounding matters. Because it enforces evidence (citations) and points out missing info, it shines on fact-checking, research, or coding tasks where correctness is critical. It is more complex and slower (requires search/tool calls), but yields highly vetted answers. Use Reflexion Agents for tasks like data lookup, code generation with static analysis, or any QA requiring references.