# Snippet Sage

This project is a Personal Knowledge Base system. It allows users to ingest text documents and ask questions about them using a Retrieval-Augmented Generation (RAG) pipeline.

## Core Features

- **Ingestion API (`/ingest`):** Accepts text content, normalizes it, splits it into chunks, generates embeddings, and stores everything in a PostgreSQL database.
- **Question-Answering API (`/answer`):** Takes a question, retrieves relevant chunks from the knowledge base using a hybrid search (lexical + vector), and uses a language model to generate a grounded answer with citations.

## Tech Stack

- **Backend:** Python, FastAPI
- **ORM:** Tortoise ORM
- **Database:** PostgreSQL with `pgvector` for vector similarity search.
- **RAG Pipeline:**
  - **Graph Orchestration:** LangGraph
  - **Prompting/Answering:** DSPy
