## MVP scope

- **Single ingestion API**: `POST /ingest`

  - Body: `{ text: string, tags?: [string], source?: string, created_at?: iso8601 }`
  - Behaviors: normalize → chunk → embed → store (Postgres with `pgvector`).
  - Rejects empty/too-short inputs; dedup by content hash.

- **Question answering API**: `GET /answer?q=…`

  - LangGraph graph: `parse → retrieve(k, hybrid) → rerank → compose answer → cite`.
  - DSPy prompt program for the answerer with a strict signature (returns `answer`, `citations[]`, `confidence`).
  - Always return citations (chunk ids + snippet previews).

- **Minimal RAG corpus**:

  - Tables: `documents`, `chunks`, `embeddings` (pgvector), `tags` (many-to-many).
  - Chunking: naive sentence/paragraph split with overlap (keep simple).

- **Observability (OTEL)**:

  - Traces for `/ingest` and `/answer` with spans: chunking, embedding, db writes/reads, vector search, LLM calls.
  - Metrics: request count/latency, tokens in/out, top retrieval hit-rates (if `eval=true` query flag set).

- **Local dev** only (no k8s/terraform): `.env`, Docker optional.

## Stack

- **Python + FastAPI + Django ORM + Postgres (pgvector) + RAG + LangGraph + DSPy + OTEL**.
