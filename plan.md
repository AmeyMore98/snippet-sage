# Project idea: “Snippet Sage — text-in, answers-out”

A tiny personal knowledge base where you **POST raw text** (notes, tickets, logs, ADRs, meeting minutes). You can later **ask questions** and get grounded answers with **citations** back to your snippets.

## Goals (why this is worth it)

- Exercise the full target stack with minimal moving parts: **Python + FastAPI + Postgres (pgvector) + RAG + LangGraph + DSPy + OTEL**.
- Keep ingestion trivial (a single endpoint) but production-ish: idempotency, basic metadata, observability.
- Practice retrieval quality, answer grounding, and prompt-programming telemetry.

## MVP scope (what you’ll actually build)

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

## Non-goals (intentionally out)

- Crawling, file uploads, PDFs, images, or multi-tenant auth.
- Complex knowledge graphs, entity resolution, or multi-hop reasoning.
- Fancy UI — this is an API-only service.

## “Just enough” quality bar

- **Answer grounding**: every answer lists 2–5 source snippets with offsets.
- **Failure modes**: graph route for “insufficient context” that returns a helpful refusal with suggestions (e.g., “try ingesting X”).
- **Basic safety**: max input size, simple API key header, rate limit (in-process).

## Stretch (only if time remains)

- **Light entity tags**: optional extraction of entities to a `entity_tags` table (still SQL, no graph DB); allow `?filter=tag:project-alpha`.
- **Reranker**: add a local cross-encoder or LLM-based re-ranking node in LangGraph.
- **Eval set**: a tiny YAML of Q→expected snippets to compute Recall\@k and grounding precision; expose `/eval/run`.

## Success criteria (clear stopping point)

- You can ingest 200–500 short snippets in under a minute.
- Typical QA round-trip < 800 ms for k≤8 (excluding LLM latency variability).
- At least 80% of evaluation questions return answers with at least one correct citation.

That’s it — tight scope, single-endpoint ingest, and a clean RAG loop that showcases **FastAPI, Postgres/pgvector, LangGraph orchestration, DSPy prompt programs, and OTEL traces**. Continuous learning with minimal yak-shaving.
