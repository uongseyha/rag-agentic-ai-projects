import os
from dotenv import load_dotenv
from langchain_ibm import ChatWatsonx
from langgraph.graph import StateGraph, END,START
from typing import TypedDict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from IPython.display import Image, display

print("Initializing the graph...")
load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key = os.getenv("OPENAI_API_KEY"),
)

def print_workflow_info(workflow, app=None):
    """Prints comprehensive information about a LangGraph workflow."""
    print("WORKFLOW INFORMATION")
    print("====================")
    print(f"Nodes: {workflow.nodes}")
    print(f"Edges: {workflow.edges}")

    
    # Use getter method for finish points if available
    try:
        finish_points = workflow.finish_points
        print(f"Finish points: {finish_points}")
    except:
        try:
            # Alternative approaches
            print(f"Finish point: {workflow._finish_point}")
        except:
            print("Finish points attribute not directly accessible")
    
    if app:
        print("\nWorkflow Visualization:")
        #display(app.get_graph().draw_png())

class ChainState(TypedDict):
    job_description: str
    resume_summary: str
    cover_letter: str

def generate_resume_summary(state: ChainState) -> ChainState:
    prompt = f"""
You're a resume assistant. Read the following job description and summarize the key qualifications and experience the ideal candidate should have, phrased as if from the perspective of a strong applicant's resume summary.

Job Description:
{state['job_description']}
"""

    response = llm.invoke(prompt)

    return {**state, "resume_summary": response.content}

def generate_cover_letter(state: ChainState) -> ChainState:
    prompt = f"""
You're a cover letter writing assistant. Using the resume summary below, write a professional and personalized cover letter for the following job.

Resume Summary:
{state['resume_summary']}

Job Description:
{state['job_description']}
"""

    response = llm.invoke(prompt)

    return {**state, "cover_letter": response.content}

workflow = StateGraph(ChainState)
workflow.add_node("generate_resume_summary", generate_resume_summary)
workflow.add_node("generate_cover_letter", generate_cover_letter)

workflow.set_entry_point("generate_resume_summary")

workflow.add_edge("generate_resume_summary", "generate_cover_letter")

workflow.set_finish_point("generate_cover_letter")

#print_workflow_info(workflow)
app = workflow.compile()

# Generate a visualization of the workflow graph
#display(Image(app.get_graph().draw_png()))

# start the workflow with an initial state
input_state = {
        "job_description": "We are looking for a data scientist with experience in machine learning, NLP, and Python. Prior work with large datasets and experience deploying models into production is required."
}

result = app.invoke(input_state)
print(result['resume_summary'])