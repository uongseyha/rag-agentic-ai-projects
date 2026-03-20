import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# system_prompt = """
# You are a helpful assistant and answer question which related to math only. 
# Otherwise, say 'sorry I can answer only math related question'.
# """

system_prompt = """You are a helpful assistant and provide the short answer to the question."""
while True:
    user_prompt = input("You: ")
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {   "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
        ]
    )
    print(response.choices[0].message.content)