# PKB MVP — Granular Step-by-Step Build Plan (tasks.md)

> Built directly from the **final architecture** (see `architecture.md`). Each task is **small**, **testable**, and focuses on **one concern**. After each task, run its test before moving on.

---

## Phase 0 — Repo & Tooling Bootstrap

### T0.1 — Initialize repository - Done

**Start:** Create new repo `pkb`.
**End:** Git initialized with Python toolchain files.
**Steps:** `git init`; add `.gitignore` (Python, venv, **pycache**, .env); add `README.md`.
**Test:** `git status` is clean; initial commit present.

### T0.2 — Dependency manifest - Done

**Start:** Empty repo.
**End:** `requirements.txt` with pinned versions.
**Steps:** Add: fastapi, uvicorn[standard], pydantic; tortoise-orm, aerich, asyncpg; opentelemetry-\* (api, sdk, exporter-otlp, instrumentation-fastapi); sentence-transformers, torch (cpu), numpy; langgraph; dspy; httpx; python-dotenv; pytest, pytest-asyncio.
**Test:** `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` completes successfully.

### T0.3 — Base directories - Done

**Start:** Repo exists.
**End:** Folder tree scaffolded.
**Steps:** Create:

```
app/{api,core,models,rag,schemas,services,tests}
scripts
```

Add `__init__.py` inside Python dirs.
**Test:** `tree` shows structure; imports don’t fail for empty packages.

---

## Phase 1 — Configuration

### T1.1 — .env and config module - Done

**Start:** No config.
**End:** `.env.example` and `app/core/config.py`.
**Steps:** Implement `Settings` via `pydantic.BaseSettings` with defaults in the architecture (ports, DB, OTEL, embedding model, K, limits). Add `DATABASE_URL`.
**Test:** `python -c "from app.core.config import Settings; print(Settings().APP_PORT)"` prints default.

---

## Phase 2 — Data Model

### T2.1 — Base models - Done

**Start:** Empty models.
**End:** `models/base.py` with `TimestampedModel` + `UUIDBase`.
**Test:** Import succeeds; `issubclass(UUIDBase, TimestampedModel)` is true.

### T2.2 — Document model - Done

**Start:** Base ready.
**End:** `models/documents.py` with fields per architecture: `source`, `user_created_at`, `content_sha256 (unique)`, `raw_text`.
**Test:** `python -c "from app.models.documents import Document; print('Document OK')"` prints OK.

### T2.3 — Chunk model - Done

**Start:** Document ready.
**End:** `models/chunks.py` with FK to Document, `ordinal`, `text`, `text_preview`, `chunk_sha256 (unique)`.
**Test:** Import succeeds; `hasattr(Chunk, 'document')`.

### T2.4 — Embedding model - Done

**Start:** Chunk ready.
**End:** `models/embeddings.py` with OneToOneField to Chunk, `vector`, `dim`, `model`.
**Test:** Import succeeds.

### T2.5 — Tag + ChunkTag - Skipped

**Start:** Above models ready.
**End:** `models/tags.py` + `models/relations.py` with unique_together `(chunk, tag)`.
**Test:** Import succeeds; `ChunkTag._meta.unique_together` is set.

---

## Phase 3 — Database, Extensions & Indices

### T3.1 — Postgres up (Docker optional) - Skipped

**Start:** No DB.
**End:** Local Postgres running.
**Steps:** Use provided `docker-compose.yml` (pgvector image).
**Test:** `psql -h localhost -U pkb -d pkb -c "select 1"` returns 1.

### T3.2 — Init SQL (pgvector + FTS + indices) - Done

**Start:** DB up.
**End:** `scripts/init_db.sql` created with statements from architecture.
**Test:** `psql -f scripts/init_db.sql` runs idempotently twice with zero errors.

### T3.3 — Tortoise ORM (Aerich) migrations - Done

**Start:** Models defined.
**End:** Migrations applied.
**Steps:** `aerich init-db` to create migration table, then `aerich migrate` to generate and apply.
**Test:** `\dt` in psql shows `documents, chunks, embeddings, tags, chunk_tags` tables.

### T3.4 — Validate generated FTS column - Done

**Start:** `chunks` table exists.
**End:** `fts tsvector` generated column present.
**Test:** `\d+ chunks` shows `fts` column; `select to_tsvector('english','hello')` sanity-checks FTS.

### T3.5 — Vector column + IVFFlat index - Done

**Start:** `embeddings` table exists.
**End:** `vector` pgvector column with dim and IVFFlat index.
**Test:** `\d+ embeddings` shows `vector`; `\di` shows `idx_embeddings_vector` with `ivfflat` method.

### T3.6 — Dedup unique indices - Done

**Start:** Tables present.
**End:** Unique on `content_sha256` and `chunk_sha256`.
**Test:** Inserting duplicate hashes raises constraint error.

---

## Phase 4 — Core Utilities

### T4.1 — Text normalization - Done

**Start:** None.
**End:** `core/text.py:normalize(text)->str`.
**Test:** Unit: collapses multi-space/newlines, normalizes unicode quotes.

### T4.2 — Preview builder - Skipped

**Start:** None.
**End:** `core/text.py:preview(text, n=160)->str`.
**Test:** Exactly ≤160 chars; no newlines; adds ellipsis when trimmed.

### T4.3 — Content hashing - Done

**Start:** None.
**End:** `core/hashing.py:sha256(text)->hex`.
**Test:** Known inputs map to expected hex; stable across calls.

---

## Phase 5 — Chunker - Done

### T5.1 — Paragraph splitter

**Start:** None.
**End:** `rag/chunker.split_paragraphs(text)->list[str]`.
**Test:** Blank-line separation yields expected paragraphs.

### T5.2 — Sentence windowing (overlap)

**Start:** Splitter ready.
**End:** `rag/chunker.window_sentences(paragraph,max_sent=6,overlap=1)`.
**Test:** For 7 sentences, returns windows with one-sentence overlap.

### T5.3 — Chunk pipeline

**Start:** Helpers ready.
**End:** `rag/chunker.make_chunks(text)->list[str]`.
**Test:** Deterministic chunks for fixed input; previews built for each.

---

## Phase 6 — Embedder - Done

### T6.1 — Model loader

**Start:** None.
**End:** `rag/embedder.load_model()` reads env and loads sentence-transformer.
**Test:** Returns model; `get_sentence_embedding_dimension()==EMBEDDING_DIM`.

### T6.2 — Batch embed

**Start:** Loader ready.
**End:** `rag/embedder.embed_texts(texts)->np.ndarray` batched with optional `batch_size`.
**Test:** Embeds 3 texts → shape `(3, dim)`; no NaNs.

### T6.3 — Persist embeddings

**Start:** Embedding OK.
**End:** Function that upserts Embedding rows for given Chunk IDs (transactional).
**Test:** After call, `select count(*) from embeddings` equals chunk count.

---

## Phase 7 — Retriever (Hybrid) - Done

### T7.1 — Vector search

**Start:** None.
**End:** `rag/retriever.vector_search(qvec, k)->[Hit]` (returns id, vscore).
**Test:** Known near-duplicate ranks top-1.

### T7.2 — Lexical FTS search

**Start:** None.
**End:** `rag/retriever.lexical_search(q, k)->[Hit]` (id, lscore).
**Test:** Query term present → positive rank; limit respected.

### T7.3 — Score normalization

**Start:** Legs ready.
**End:** `rag/retriever.normalize_scores(hits)->[Hit]` (min-max per leg).
**Test:** All scores ∈ [0,1]; monotonic ordering preserved.

### T7.4 — Fusion

**Start:** Normalizers ready.
**End:** `rag/retriever.fuse(vhits, lhits, alpha=0.6, k=12)->[Hit]`.
**Test:** Synthetic scores produce expected ordering and truncation to k.

### T7.5 — Optional tag filter

**Start:** Fusion works.
**End:** `rag/retriever.filter_by_tags(hits, tags)->[Hit]`.
**Test:** Passing tags reduces set appropriately.

---

## Phase 8 — Reranker - Done

### T8.1 — Interface + stub

**Start:** None.
**End:** `rag/reranker.score(query,text)->float` (stub: token overlap).
**Test:** Higher overlap yields higher score.

### T8.2 — Top-N selection

**Start:** Stub ready.
**End:** `rag/reranker.topn(query, candidates, n=6)->list` sorted by score.
**Test:** Returns exactly n items (or fewer if not enough).

_(Optional later: swap stub with small cross-encoder.)_ - Done

---

## Phase 9 — DSPy Answerer - Done

### T9.1 — Signature

**Start:** dspy installed.
**End:** `rag/dspy_program.py:AnswerSignature` with fields (question, context, answer, citations, confidence).
**Test:** Import passes; reflection shows fields.

### T9.2 — Program

**Start:** Signature ready.
**End:** `Answerer = dspy.Program(AnswerSignature)`.
**Test:** Calling with dummy context returns outputs (possibly mocked).

### T9.3 — Contract wrapper

**Start:** Program ready.
**End:** `rag/dspy_program.run_answerer(question, passages)->dict`.
**Test:** `confidence` clamped to [0,1]; `citations` subset of passage IDs.

---

## Phase 10 — LangGraph Graph - Done

### T10.1 — Parse node

**Start:** None.
**End:** `rag/graph.parse_node(state)->state` (normalize/trim question).
**Test:** State returns `q_norm` differing only by normalization.

### T10.2 — Retrieve node

**Start:** Retriever ready.
**End:** `rag/graph.retrieve_node(state)->adds state['hits']`.
**Test:** Hits length ≤ k; each has id and score fields.

### T10.3 — Rerank node

**Start:** Reranker ready.
**End:** `rag/graph.rerank_node(state)->state['top_passages']`.
**Test:** Top passages sorted by rerank score desc; contains texts.

### T10.4 — Answer node

**Start:** DSPy wrapper ready.
**End:** `rag/graph.answer_node(state)->state['answer_bundle']`.
**Test:** Bundle includes `answer`, `citations`, `confidence` keys.

### T10.5 — Cite node

**Start:** Answer exists.
**End:** `rag/graph.cite_node(state)->state['response']` with `{chunk_id, preview, score}`.
**Test:** Previews are ≤160 chars; scores present.

### T10.6 — Compile graph

**Start:** Nodes ready.
**End:** `rag/graph.build_graph()->callable`.
**Test:** Running with mocked retriever produces full response payload.

---

## Phase 11 — Services - Done

### T11.1 — Ingestion service

**Start:** Core utils & models ready.
**End:** `services/ingestion_service.ingest(payload)->(doc_id, chunk_ids)` orchestrating normalize→hash→dedup→chunk→embed→store→tag.
**Test:** Returns IDs; dedup triggers on second call with same text.

### T11.2 — QA service

**Start:** Graph ready.
**End:** `services/qa_service.answer(q,k,eval_mode)->dict`.
**Test:** With seeded corpus, returns non-empty `citations` and a float `confidence`.

---

## Phase 12 — FastAPI Surface - Done

### T12.1 — App factory

**Start:** None.
**End:** `app/main.py` creates FastAPI app, mounts routers, inits OTEL, calls `init_tortoise()`.
**Test:** `uvicorn app.main:app --reload` serves `/docs` (OpenAPI loads).

### T12.2 — Ingest schema

**Start:** None.
**End:** `schemas/ingest.py` with Pydantic models + min-length validation.
**Test:** POST invalid payload returns 422 (FastAPI validation).

### T12.3 — Answer schema

**Start:** None.
**End:** `schemas/answer.py` for response contract with citations array.
**Test:** GET empty `q` returns 422; valid query returns typed payload.

### T12.4 — POST /ingest route

**Start:** Service ready.
**End:** `api/ingest.py` implemented with spans/metrics.
**Test:** `curl` returns 201 with `{document_id, chunk_ids[], chunk_count, tags[]}`.

### T12.5 — GET /answer route

**Start:** QA service ready.
**End:** `api/answer.py` implemented with `k` and `eval` query params.
**Test:** `curl '.../answer?q=test'` returns `{answer,citations,confidence}`; `k` capped to ≤50.

---

## Phase 13 — Observability (OTEL)

### T13.1 — Tracer & meter init

**Start:** None.
**End:** `core/otel.py` sets tracer/meter, console exporter default.
**Test:** Starting app prints resource + spans on requests.

### T13.2 — Auto-instrumentation

**Start:** Tracer exists.
**End:** Instrument FastAPI/ASGI, asyncpg, httpx.
**Test:** DB and HTTP spans appear in console exporter.

### T13.3 — Ingestion spans

**Start:** None.
**End:** Add spans: `normalize`, `hash`, `chunking`, `embedding(batch)`, `db.write.*`.
**Test:** `/ingest` trace shows child spans with timings.

### T13.4 — Answering spans

**Start:** None.
**End:** Add spans: `parse`, `retrieve.vector`, `retrieve.lexical`, `retrieve.fuse`, `rerank`, `answer.dspy`, `cite`, `db.read.*`.
**Test:** `/answer` trace shows sequence per architecture.

### T13.5 — Metrics

**Start:** Meter exists.
**End:** Counters: `ingest.requests`, `answer.requests`; histograms: `*.latency_ms`; token metrics in answer node.
**Test:** Repeated requests increment counters; histograms populated.

### T13.6 — Eval-mode retrieval metrics

**Start:** None.
**End:** If `eval=true`, compute `retrieval.hit_rate_at_k`, `mean_score_top1`, `mean_score_topk`.
**Test:** Toggle flag and observe metric emission (console/OTLP).

---

## Phase 14 — Tests

### T14.1 — Pytest scaffolding

**Start:** None.
**End:** `tests/conftest.py` with test DB/schema fixture, FastAPI test client.
**Test:** `pytest -q` runs 0 tests green.

### T14.2 — Core utils tests

**Start:** Utils ready.
**End:** Tests for normalize, preview, sha256.
**Test:** Green.

### T14.3 — Chunker tests

**Start:** Chunker ready.
**End:** Tests: paragraph split, sentence windowing, pipeline determinism.
**Test:** Green.

### T14.4 — Embedder tests

**Start:** Embedder ready.
**End:** Deterministic shapes; no NaNs; spans recorded (mock tracer).
**Test:** Green.

### T14.5 — Persistence integration test

**Start:** Models & DB ready.
**End:** Ingest pipeline writes doc, chunks, embeddings, tags atomically.
**Test:** Counts match; rolling back on simulated failure keeps DB consistent.

### T14.6 — Hybrid retrieval tests

**Start:** Retriever ready.
**End:** Seed two docs; expecting relevant chunk ranks top after fusion.
**Test:** Assert ordering.

### T14.7 — Reranker tests

**Start:** Reranker ready.
**End:** Token-overlap stub monotonic; topn returns n.
**Test:** Green.

### T14.8 — DSPy wrapper tests

**Start:** Wrapper ready.
**End:** Confidence clamp; citations subset.
**Test:** Green.

### T14.9 — LangGraph path test

**Start:** Graph ready.
**End:** Mock retriever/reranker/answerer; ensure state flows parse→retrieve→rerank→answer→cite.
**Test:** Response schema valid.

### T14.10 — API tests

**Start:** Endpoints ready.
**End:** `/ingest`: 201 + schema; duplicate: 409; too short: 422. `/answer`: valid payload with citations; zero-corpus returns empty answer and 0.0 confidence.
**Test:** Green.

---

## Phase 15 — Local Ops & Smoke

### T15.1 — Run server

**Start:** App compiled.
**End:** Dev server running with reload.
**Test:** Open `/docs` in browser; endpoints visible.

### T15.2 — Smoke ingestion

**Start:** Server up.
**End:** One document ingested.
**Test:** CURL from architecture returns 201; DB shows chunk+embedding counts.

### T15.3 — Smoke answer

**Start:** Content present.
**End:** One answer with citations.
**Test:** CURL returns non-empty `citations[]` with previews.

---

## Phase 16 — Hardening (MVP)

### T16.1 — Input caps & sanitation

**Start:** None.
**End:** Enforce `MAX_INPUT_CHARS`; strip control chars.
**Test:** Oversized payload rejected with 422; control chars removed.

### T16.2 — Transaction boundaries

**Start:** Partial writes possible.
**End:** Ingestion runs in atomic transaction; partial failures handled.
**Test:** Simulated embedding error → rollback or partial-save strategy validated.

### T16.3 — Guardrails on answer params

**Start:** None.
**End:** Cap `k≤50`; sanitize query whitespace.
**Test:** `k=500` coerced to 50; extra spaces trimmed.

---

## Quick Index

- **APIs:** T12.4, T12.5
- **Models/DB:** T2.1–T3.6
- **RAG:** T5.1–T10.6
- **OTEL:** T13.1–T13.6
- **Tests:** T14.1–T14.10
- **Ops:** T15.1–T15.3
