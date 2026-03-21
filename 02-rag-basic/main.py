import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from qdrant_client import QdrantClient

load_dotenv()

# Ensure indexing has been completed
qdrant_client = QdrantClient(url="http://localhost:6333")
collection_names = [c.name for c in qdrant_client.get_collections().collections]

if "rag-basic" not in collection_names:
    print("Collection not found. Running index_data.py first...")
    index_script = Path(__file__).parent / "index_data.py"
    subprocess.run([sys.executable, str(index_script)], check=True)
    print("Indexing complete. Proceeding with query...\n")

openai_client = OpenAI()

# Vector Embeddings
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="rag-basic",
    embedding=embedding_model,
)

# Take user input
user_query = input("Ask something: ")

# Relevant chunks from the vector db
search_results = vector_db.similarity_search(query=user_query)
context = "\n\n\n".join([f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}" for result in search_results])

SYSTEM_PROMPT = f"""
 You are a helpfull AI Assistant who answeres user query based on the available context retrieved from a PDF file along with page_contents and page number.
 You should only ans the user based on the following context and navigate the user to open the right page number to know more.
 Context:
 {context}
"""

response = openai_client.chat.completions.create(
    model="gpt-5",
    messages=[
        { "role": "system", "content":SYSTEM_PROMPT  },
        { "role": "user", "content":user_query  },
    ]
)

print(f"🤖: {response.choices[0].message.content}")