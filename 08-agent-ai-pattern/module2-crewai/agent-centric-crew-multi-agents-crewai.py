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

agent_centric_agent = Agent(
    role="The Daily Dish Inquiry Specialist",
    goal="""Accurately answer customer questions about The Daily Dish restaurant. 
    You must decide whether to use the restaurant's FAQ PDF or a web search to find the best answer.""",
    backstory="""You are an AI assistant for 'The Daily Dish'.
    You have access to two tools: one for searching the restaurant's FAQ document and another for searching the web.
    Your job is to analyze the user's question and choose the most appropriate tool to find the information needed to provide a helpful response.""",
    tools=[pdf_search_tool, web_search_tool],
    verbose=True,
    allow_delegation=False,
    llm=llm
)

agent_centric_task = Task(
    description="Answer the following customer query: '{customer_query}'. "
                "Analyze the question and use the tools at your disposal (PDF search or web search) to find the most relevant information. "
                "Synthesize the findings into a clear and friendly response.",
    expected_output="A comprehensive and well-formatted answer to the customer's query.",
    agent=agent_centric_agent
)

agent_centric_crew = Crew(
    agents=[agent_centric_agent],
    tasks=[agent_centric_task],
    process=Process.sequential,
    verbose=False
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
        result_agent_centric = agent_centric_crew.kickoff(inputs={'customer_query': user_input})
        print("\n--- The Daily Dish Assistant ---")
        print(result_agent_centric)
        print("--------------------------------")
    except Exception as e:
        print(f"An error occurred: {e}")

