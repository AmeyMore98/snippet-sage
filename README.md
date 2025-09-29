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

## Getting Started

### Prerequisites

- Python 3.12
- PostgreSQL 16+ with the `pgvector` extension enabled.
- An API key for a DSPy-compatible language model.

### Configuration

1.  **Create a `.env` file** in the project root directory. You can copy the example:

    ```bash
    cp .env.example .env
    ```

2.  **Set the environment variables** in the `.env` file:

    - `DATABASE_URL`: The connection string for your PostgreSQL database.
      - **Example:** `postgres://user:password@localhost:5432/snippet_sage`
    - `DSPY_MODEL`: The LM to use with DSPy
    - `DSPY_LM_API_KEY`: Your API key for the LM.

## Running the Application

### Local Development

To run the application with live reloading, use `uvicorn`:

```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

### Production

The included `Dockerfile` uses `gunicorn` for production:

```bash
gunicorn -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8080 app.main:app
```

## Testing

To run the test suite, use `pytest`:

```bash
pytest
```

## Project Structure

```
.
├── app/
│   ├── api/          # FastAPI routers and endpoints
│   ├── core/         # Configuration and core utilities
│   ├── models/       # Tortoise ORM models
│   ├── rag/          # RAG pipeline components (chunker, embedder, etc.)
│   ├── schemas/      # Pydantic schemas for API requests/responses
│   ├── services/     # Business logic for ingestion and QA
│   └── tests/        # Pytest tests
├── migrations/       # Aerich database migrations
├── .env.example      # Example environment variables
├── requirements.txt  # Project dependencies
└── README.md         # This file
```

## API Endpoints

### Ingestion

- **Endpoint:** `POST /api/v1/ingest`
- **Description:** Ingests a document into the knowledge base.
- **Request Body:**
  ```json
  {
    "text": "The content of the document to ingest."
  }
  ```
- **Response:**
  ```json
  {
    "document_id": "doc_123",
    "chunk_ids": ["chunk_abc", "chunk_def"],
    "chunk_count": 2
  }
  ```

### Question Answering

- **Endpoint:** `POST /api/v1/answer`
- **Description:** Answers a question based on the ingested documents.
- **Request Body:**
  ```json
  {
    "q": "What is the main topic of the document?",
    "k": 3
  }
  ```
- **Response:**
  ```json
  {
    "answer": "The main topic is...",
    "hits": [
      {
        "id": "chunk_abc",
        "score": 0.89,
        "text": "The relevant text snippet..."
      }
    ]
  }
  ```
