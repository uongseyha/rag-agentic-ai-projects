import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_ibm import ChatWatsonx
from typing import TypedDict, Optional

load_dotenv()

openai_llm = ChatOpenAI(
    model="gpt-4.1-nano",
    api_key = os.getenv("OPENAI_API_KEY"),
)
watsonx_llm = ChatWatsonx(
    model_id="ibm/granite-guardian-3-8b",
    url="https://us-south.ml.cloud.ibm.com",
    project_id=os.getenv("WATSONX_PROJECT_ID"),
    api_key=os.getenv("WATSONX_API_KEY"),
)

class AuthState(TypedDict):
    username: Optional[str] 
    password: Optional[str]
    is_authenticated: Optional[bool]
    output: Optional[str]

# auth_state_1: AuthState = {
#     "username": "alice123",
#     "password": "123",
#     "is_authenticated": True,
#     "output": "Login successful."
# }
#print(f"auth_state_1: {auth_state_1}")

# auth_state_2: AuthState = {
#     "username":"",
#     "password": "wrongpassword",
#     "is_authenticated": False,
#     "output": "Authentication failed. Please try again."
# }
#print(f"auth_state_2: {auth_state_2}")

def input_node(state):
    print(state)
    if state.get('username', "") =="":
        username = input("What is your username?")

    password = input("Enter your password: ")
    
    if state.get('username', "") =="":
        return {"username":username, "password":password}
    else:
        return {"password": password}
    
#input_node(auth_state_1)
#input_node(auth_state_2)

def validate_credentials_node(state):
    # Extract username and password from the state
    username = state.get("username", "")
    password = state.get("password", "")

    print("Username :", username, "Password :", password)
    # Simulated credential validation
    if username == "test_user" and password == "secure_password":
        is_authenticated = True
    else:
        is_authenticated = False

    # Return the updated state with authentication result
    return {"is_authenticated": is_authenticated}

# Define the success node
def success_node(state):
    return {"output": "Authentication successful! Welcome."}

# Define the failure node
def failure_node(state):
    return {"output": "Not Successful, please try again!"}

def router(state):
    if state['is_authenticated']:
        return "success_node"
    else:
        return "failure_node"
    
# Create an instance of StateGraph with the GraphState structure
workflow = StateGraph(AuthState)
workflow.add_node("InputNode", input_node)
workflow.add_node("ValidateCredential", validate_credentials_node)
workflow.add_node("SuccessNode", success_node)
workflow.add_node("FailureNode", failure_node)
workflow.add_edge("InputNode", "ValidateCredential")
workflow.add_edge("SuccessNode", END)
workflow.add_edge("FailureNode", "InputNode")
workflow.add_conditional_edges("ValidateCredential", router, {"success_node": "SuccessNode", "failure_node": "FailureNode"})
workflow.set_entry_point("InputNode")
app = workflow.compile()
inputs = {"username": "test_user"}
result = app.invoke(inputs)
#print(result)
#result['output']