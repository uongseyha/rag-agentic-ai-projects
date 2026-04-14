import os
from dotenv import load_dotenv
import json
import getpass
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessageGraph
from langchain_ibm import ChatWatsonx

load_dotenv()

openai_llm = ChatOpenAI(
    model="gpt-4.1-nano",
    api_key = "your openai api key here",
)
watsonx_llm = ChatWatsonx(
    model_id="ibm/granite-3-2-8b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="your project id associated with the API key",
    api_key="your watsonx.ai api key here",
)