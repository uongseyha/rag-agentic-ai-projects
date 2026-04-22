import os
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from crewai import LLM,Agent,Task,Crew, Process

load_dotenv()
os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')

search_tool=SerperDevTool()
#print(type(search_tool))

search_query = "Latest Breakthroughs in machine learning"
search_results = search_tool.run(search_query=search_query)

# Print the results
print(f"Search Results for '{search_query}':\n")
#print(search_results)

#print("keys of search_results", search_results.keys())

llm = LLM(
        model="gpt-4o-mini",  # Changed to OpenAI GPT-4o
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=2000,
)

research_agent = Agent(
  role='Senior Research Analyst',
  goal='Uncover cutting-edge information and insights on any subject with comprehensive analysis',
  backstory="""You are an expert researcher with extensive experience in gathering, analyzing, and synthesizing information across multiple domains. 
  Your analytical skills allow you to quickly identify key trends, separate fact from opinion, and produce insightful reports on any topic. 
  You excel at finding reliable sources and extracting valuable information efficiently.""",
  verbose=True,
  allow_delegation=False,
  llm = llm,
  tools=[SerperDevTool()]
)

#print(research_agent)

# Define your agents with roles and goals
# Define the Writer Agent
writer_agent = Agent(
  role='Tech Content Strategist',
  goal='Craft well-structured and engaging content based on research findings',
  backstory="""You are a skilled content strategist known for translating 
  complex topics into clear and compelling narratives. Your writing makes 
  information accessible and engaging for a wide audience.""",
  verbose=True,
  llm = llm,
  allow_delegation=True
)

#writer_agent 

social_agent = Agent(
    role='Social Media Strategist',
    goal='Generate engaging social media snippets based on the full article',
    backstory="A digital storyteller who excels at crafting compelling posts to drive engagement and traffic.",
    verbose=True
)

research_task = Task(
  description="Analyze the major {topic}, identifying key trends and technologies. Provide a detailed report on their potential impact.",
  agent=research_agent,
  expected_output="A detailed report on {topic}, including trends, emerging technologies, and their impact."
)

# Create a task for the Writer Agent
writer_task = Task(
  description="Create an engaging blog post based on the research findings about {topic}. Tailor the content for a tech-savvy audience, ensuring clarity and interest.",
  agent=writer_agent,
  expected_output="A 4-paragraph blog post on {topic}, written clearly and engagingly for tech enthusiasts."
)

social_task = Task(
    description=(
        "Summarize the blog post about {topic} into 2–3 engaging social media posts "
        "suitable for platforms like LinkedIn or Twitter. Make sure the tone is informative, "
        "professional, and encourages further reading."
    ),
    agent=social_agent,
    expected_output="A series of 2–3 well-written social posts highlighting the key insights from the blog content."
)

crew = Crew(
    agents=[research_agent, writer_agent, social_agent],
    tasks=[research_task, writer_task, social_task],
    process=Process.sequential,
    verbose=True 
)

result = crew.kickoff(inputs={"topic": "Latest top 3 Generative AI breakthroughs"})
#print("Final Result:\n", result)
#type(result)
final_output = result.raw
print("Final output:", final_output)
tasks_outputs = result.tasks_output

print("Writer task description:", tasks_outputs[1].description)
print(" \nOutput of writer task:", tasks_outputs[1].raw)

print("We can get the agent for researcher task:  ",tasks_outputs[0].agent)
print("We can get the agent for the writer task: ",tasks_outputs[1].agent)
print("We can get the agent for the social task: ",tasks_outputs[2].agent)


# token_count = result.token_usage.total_tokens
# prompt_tokens = result.token_usage.prompt_tokens
# completion_tokens = result.token_usage.completion_tokens

# print(f"Total tokens used: {token_count}")
# print(f"Prompt tokens: {prompt_tokens} (used for instructions to the model)")
# print(f"Completion tokens: {completion_tokens} (generated in response)")