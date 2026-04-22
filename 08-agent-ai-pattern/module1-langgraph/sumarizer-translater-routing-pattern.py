import os
from dotenv import load_dotenv
from langchain_ibm import ChatWatsonx
from langgraph.graph import StateGraph, END,START
from typing import TypedDict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from IPython.display import Image, display

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key = os.getenv("OPENAI_API_KEY"),
)

class RouterState(TypedDict):
    user_input: str
    task_type: str
    output: str

class Router(BaseModel):
    role: str = Field(..., description="Decide whether the user wants to summarize a passage  ouput 'summarize'  or translate text into French oupput translate.")

llm_router=llm.bind_tools([Router])
response=llm_router.invoke("summarize this I love the sun its so warm")

def router_node(state: RouterState) -> RouterState:
    routing_prompt = f"""
    You are an AI task classifier.
    
    Decide whether the user wants to:
    - "summarize" a passage
    - or "translate" text into French
    
    Respond with just one word: 'summarize' or 'translate'.
    
    User Input: "{state['user_input']}"
    """

    response = llm_router.invoke(routing_prompt)

    return {**state, "task_type": response.tool_calls[0]['args']['role']} # This becomes the next node's name!

def router(state: RouterState) -> str:
    return state['task_type']

def summarize_node(state: RouterState) -> RouterState:
    prompt = f"Please summarize the following passage:\n\n{state['user_input']}"
    response = llm.invoke(prompt)
    
    return {**state, "task_type": "summarize", "output": response.content}

def translate_node(state: RouterState) -> RouterState:
    prompt = f"Translate the following text to French:\n\n{state['user_input']}"
    response = llm.invoke(prompt)

    return {**state, "task_type": "translate", "output": response.content}

workflow = StateGraph(RouterState)
workflow.add_node("router", router_node)
workflow.add_node("summarize", summarize_node)
workflow.add_node("translate", translate_node)

workflow.set_entry_point("router")

workflow.add_conditional_edges("router", router, {
    "summarize": "summarize",
    "translate": "translate"
})

workflow.set_finish_point("summarize")
workflow.set_finish_point("translate")

app = workflow.compile()

#init start
input_text = {
        #"user_input": "Can you translate this sentence: I love programming?"
        "user_input": "Can you summarize the plot of the movie Inception?"
    }

result = app.invoke(input_text)

print(result['output'])
print(result['task_type'])