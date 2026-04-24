import os
from crewai_tools import PDFSearchTool, SerperDevTool
from dotenv import load_dotenv
from crewai import LLM,Agent,Task,Crew, Process
from pydantic import Field, BaseModel
from typing import List, Dict, Optional
from IPython.display import display, JSON, Markdown
from datetime import datetime
from leftover import LeftoversCrew
import warnings
warnings.filterwarnings('ignore') #Keeps Jupyter Notebook clean (not part of functionality)

load_dotenv()
os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')

web_search_tool = SerperDevTool()
llm = LLM(
        model="gpt-4o-mini",  # Changed to OpenAI GPT-4o
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=2000,
)

#print("LLM and Search Tool initialized successfully.")

pdf_search_tool = PDFSearchTool(
    pdf="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/7vgNfis17dQfjHAiIKkBOg/The-Daily-Dish-FAQ.pdf",
    config=dict(
        embedder=dict(
            provider="huggingface",
            config=dict(
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
        )
    )
)

task_centric_agent = Agent(
    role="Customer Service Specialist",
    goal="Provide exceptional customer service by following a multi-step process to answer customer questions accurately.",
    backstory="""You are an AI assistant for 'The Daily Dish'.
    You are an expert at following instructions. You will be given a sequence of tasks to complete.
    For each task, you will be provided with the specific tool needed to accomplish it.
    Your job is to execute each task diligently and pass the results to the next step.""",
    tools=[], # The agent is not given any tools directly
    verbose=True,
    allow_delegation=False,
    llm=llm
)

faq_search_task = Task(
    description="Search the restaurant's FAQ PDF for information related to the customer's query: '{customer_query}'.",
    expected_output="A snippet of the most relevant information from the PDF, or a statement that the information was not found.",
    tools=[pdf_search_tool], # Tool assigned directly to the task
    agent=task_centric_agent
)

response_drafting_task = Task(
    description="Using the information gathered from the FAQ search, draft a friendly and comprehensive response to the customer's query: '{customer_query}'.",
    expected_output="The final, customer-facing response.",
    agent=task_centric_agent,
    context=[faq_search_task]
)

task_centric_crew = Crew(
    agents=[task_centric_agent],
    tasks=[faq_search_task, response_drafting_task],
    process=Process.sequential,
    verbose=True
)

print("\nWelcome to The Daily Dish Chatbot!")
print("What would you like to know? (Type 'exit' to quit)")

while True: 
    user_input = input("\nYour question: ").lower()
    if user_input == 'exit':
        print("Thank you for chatting. Have a great day!")
        break
    
    if not user_input:
        print("Please type a question.")
        continue

    try:
        # Here we use our more advanced, task-centric crew
        result_task_centric = task_centric_crew.kickoff(inputs={'customer_query': user_input})
        print("\n--- The Daily Dish Assistant ---")
        print(result_task_centric)
        print("--------------------------------")
    except Exception as e:
        print(f"An error occurred: {e}")