import os
from dotenv import load_dotenv
from crewai import LLM, Agent, Task, Crew
from crewai.tools import tool
from functools import reduce
import re

load_dotenv()

llm = LLM(
        model="gpt-4o-mini",  # Changed to OpenAI GPT-4o
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=2000,
)

@tool("Add Two Numbers Tool")
def add_numbers(data: str) -> int:
    """
    Extracts and adds integers from the input string.
    Example input: 'add 1 and 2' or '[1,2,3,4]'
    Output: sum of the numbers
    """
    # Find all integers in the string
    numbers = list(map(int, re.findall(r'-?\d+', data)))
    return sum(numbers)

from functools import reduce

@tool("Multiply Numbers Tool")
def multiply_numbers(data: str) -> int:
    """
    Extracts and multiplies integers from the input string.
    Example input: 'multiply 2 and 3' or '[2,3,4]'
    Output: the product of all numbers found
    """
    numbers = list(map(int, re.findall(r'-?\d+', data)))
    return reduce(lambda x, y: x * y, numbers, 1)

calculator_agent = Agent(
    role="Calculator",
    goal="Extracts, adds, or multiplies numbers when asked, using the Add Two Numbers and Multiply Numbers tools.",
    backstory="An expert at parsing numeric instructions and computing sums or products.",
    tools=[add_numbers, multiply_numbers],
    llm=llm,
    allow_delegation=False
)

calculation_task = Task(
    description="Extract numbers from '{numbers}' and either add or multiply them, depending on the natural-language instruction.",
    expected_output="An integer result (sum or product) based on the user’s request.",
    agent=calculator_agent
)

crew = Crew(
    agents=[calculator_agent],
    tasks=[calculation_task],
    # verbose=True #Uncomment this to see the steps taken to get the final answer
)

# Inputs for addition…
result = crew.kickoff(inputs={'numbers': 'please add 4, 5, and 6'})
print("Sum result:", result)

# Inputs for multiplication…
result = crew.kickoff(inputs={'numbers': 'multiply 7 and 8 also 9 dont forget 10'})
print("Product result:", result)