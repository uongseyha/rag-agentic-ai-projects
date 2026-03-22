# RAG & Agentic AI Projects

A collection of hands-on projects exploring LLMs, RAG pipelines, and agentic AI patterns.

## Projects

### [01-chatbot](01-chatbot/)

Simple Q&A chatbots using OpenAI and Google Gemini APIs. Includes single-turn question answering with both providers and a multi-turn chat loop using Gemini via the OpenAI-compatible endpoint.

**Tech:** Python, OpenAI API, Google Gemini API, python-dotenv

---

### [02-rag-basic](02-rag-basic/)

A Retrieval-Augmented Generation (RAG) pipeline that ingests a PDF document, stores vector embeddings in Qdrant, and answers user questions grounded in the document content with page number references.

**Tech:** Python, OpenAI GPT-5, text-embedding-3-large, Qdrant, LangChain, Docker Compose

---

### [03-gradio-rag-chatbot](03-gradio-rag-chatbot/)

An interactive RAG chatbot with a Gradio interface where users upload a PDF and ask questions, and the system retrieves relevant chunks to generate grounded answers.

**Tech:** Python, Gradio, IBM watsonx (Mistral Medium + Slate Embeddings), LangChain, ChromaDB, python-dotenv
