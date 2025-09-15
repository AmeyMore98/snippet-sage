
# Personal Knowledge Base (PKB) — MVP Architecture

> **Scope + Stack (as requested)**
>
> - **APIs**: `POST /ingest`, `GET /answer?q=…`
> - **Behavior**: normalize → chunk → embed → store (Postgres + `pgvector`); QA via **LangGraph**: parse → retrieve(k, hybrid) → rerank → compose answer → cite; **DSPy** prompt program for answerer with strict signature.
> - **Corpus**: `documents`, `chunks`, `embeddings` (pgvector), `tags` (+ M2M).
> - **Observability**: OpenTelemetry traces & metrics.
> - **Local dev only**: `.env`, Docker optional.
>
> - **Stack**: **Python + FastAPI + Tortoise ORM + Postgres (pgvector) + RAG + LangGraph + DSPy + OTEL**.

---

## 1) Repository & Runtime Layout

```text
pkb/
├─ app/
│  ├─ main.py                      # FastAPI app factory; mounts routers; OTEL init
│  ├─ api/
│  │  ├─ ingest.py                 # POST /ingest
│  │  └─ answer.py                 # GET /answer
│  ├─ core/
│  │  ├─ config.py                 # Env, settings, constants
│  │  ├─ otel.py                   # OTEL tracer, meter, instrumentations
│  │  ├─ db.py                     # Tortoise ORM bootstrap
│  │  ├─ hashing.py                # Content hashing & dedup helpers
│  │  ├─ text.py                   # Normalization, simple sentence/para split
│  ├─ models/
│  │  ├─ __init__.py               # Tortoise ORM model definitions
│  │  ├─ base.py                   # Abstract TimestampedModel, UUIDBase
│  │  ├─ documents.py              # Document model
│  │  ├─ chunks.py                 # Chunk model (+ FK to Document)
│  │  ├─ embeddings.py             # Embedding model (pgvector)
│  │  ├─ tags.py                   # Tag model
│  │  └─ relations.py              # ChunkTag through table
│  ├─ rag/
│  │  ├─ chunker.py                # Naive split w/ overlap
│  │  ├─ embedder.py               # Wrapper over embedding model (local or API)
│  │  ├─ retriever.py              # Hybrid (BM25 via Postgres FTS + vector)
│  │  ├─ reranker.py               # Lightweight cross-encoder or LLM score
│  │  ├─ citations.py              # Format {id, preview, score}
│  │  ├─ graph.py                  # LangGraph assembly: parse → retrieve → rerank → answer → cite
│  │  └─ dspy_program.py           # DSPy Signature + program for answerer
│  ├─ schemas/
│  │  ├─ ingest.py                 # Pydantic request/response models
│  │  └─ answer.py
│  ├─ services/
│  │  ├─ ingestion_service.py      # Orchestrates normalize→chunk→embed→store + traces
│  │  ├─ qa_service.py             # Orchestrates question answering graph execution
│  │  ├─ metrics.py                # Custom metrics emission (requests, tokens, hit-rate)
│  │  └─ eval.py                   # Optional eval helpers (hit-rate if ?eval=true)
│  └─ tests/                       # Lightweight pytest
├─ scripts/
│  └─ init_db.sql                  # CREATE EXTENSION, FTS, indices
├─ aerich.ini                      # Aerich migration tool config
├─ migrations/                     # Aerich-generated migrations
├─ .env.example
├─ docker-compose.yml              # Optional: Postgres + pgvector
├─ Dockerfile                      # Optional: app container
└─ README.md
```

### What each part does & how they connect

- `FastAPI (app/main.py)` is the HTTP surface. It:
  - boots **Tortoise ORM**,
  - registers routers (`/ingest`, `/answer`),
  - initializes OpenTelemetry tracing/metrics,
  - injects **tracer** & **meter** into request state for span/metric emission.

- `core/db.py` bridges FastAPI to **Tortoise ORM** for models, migrations (via Aerich), and transactions. Using Tortoise here gives:
  - a modern, async ORM.
  - `aerich` for database migrations.

- `models/*` define the **Minimal RAG corpus** (see §3).

- `rag/*` is the RAG toolkit:
  - `chunker.py` implements sentence/paragraph split with configurable overlap.
  - `embedder.py` computes vector embeddings (e.g., OpenAI, HuggingFace, or local).
  - `retriever.py` mixes **pgvector** ANN search with **Postgres FTS** (hybrid).
  - `reranker.py` reranks top-k (either a tiny cross-encoder or LLM re-score).
  - `graph.py` wires a **LangGraph** graph: `parse → retrieve → rerank → compose → cite`.
  - `dspy_program.py` expresses the **DSPy** program + strict signature for the answerer.

- `services/ingestion_service.py` and `services/qa_service.py` orchestrate domain logic and emit **OTEL** spans/metrics at each step.

- `schemas/*` holds Pydantic request/response contracts.

- `scripts/init_db.sql` installs **pgvector**, creates indexes, FTS config.

- `docker-compose.yml` (optional) spins up Postgres with `pgvector`.

---

## 2) Configuration & State

### Environment (.env)
```
# Server
APP_ENV=dev
APP_HOST=0.0.0.0
APP_PORT=8000

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=pkb
POSTGRES_USER=pkb
POSTGRES_PASSWORD=pkb
# Full DB URL for Tortoise
DATABASE_URL=postgres://pkb:pkb@localhost:5432/pkb

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIM=384

# OTEL
OTEL_SERVICE_NAME=pkb-mvp
OTEL_EXPORTER=console        # or otlp
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# RAG
RETRIEVAL_K=12
CHUNK_OVERLAP_SENTENCES=1
MIN_INPUT_CHARS=40
MAX_INPUT_CHARS=200_000
```

### Where **state** lives

- **Postgres** is the source of truth:
  - Raw document rows, chunk rows, embeddings (vector column), tags & relations.
  - **Dedup** is enforced by a unique index on `content_sha256` (document-level) and `chunk_sha256` (chunk-level).
- **Transient** state in memory:
  - Request-scoped spans, counters, token counts, and temporary graph node outputs.
- **No file/object storage** is needed for MVP (text-only ingestion).

---

## 3) Data Model (Tortoise ORM ↔ Postgres)

> Minimal corpus: `documents`, `chunks`, `embeddings` (pgvector), `tags` (+ M2M join)

```python
# app/models/base.py
from tortoise import fields, models

class TimestampedModel(models.Model):
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        abstract = True

class UUIDBase(TimestampedModel):
    id = fields.UUIDField(pk=True)

    class Meta:
        abstract = True
```

```python
# app/models/documents.py
from tortoise import fields
from .base import UUIDBase

class Document(UUIDBase):
    source = fields.CharField(max_length=255, null=True)   # e.g. "web", "cli", "note"
    user_created_at = fields.DatetimeField(null=True)      # from request body
    content_sha256 = fields.CharField(max_length=64, unique=True)      # dedup
    raw_text = fields.TextField()                                      # normalized

    class Meta:
        table = "documents"
```

```python
# app/models/chunks.py
from tortoise import fields
from .base import UUIDBase

class Chunk(UUIDBase):
    document = fields.ForeignKeyField("models.Document", related_name="chunks", on_delete=fields.CASCADE)
    ordinal = fields.IntField()                                    # position within document
    text = fields.TextField()
    text_preview = fields.CharField(max_length=256)                    # UI/debug
    chunk_sha256 = fields.CharField(max_length=64, unique=True)

    class Meta:
        table = "chunks"
```

```python
# app/models/embeddings.py
# Note: for pgvector, a custom field is needed. See tortoise-orm-pgvector.
# For simplicity, we'll represent it as a placeholder.
from tortoise import fields
from .base import UUIDBase

# from tortoise_orm_pgvector.fields import VectorField

class Embedding(UUIDBase):
    chunk = fields.OneToOneField("models.Chunk", on_delete=fields.CASCADE, related_name="embedding")
    # vector = VectorField(dimensions=384) # Example with tortoise-orm-pgvector
    vector = fields.BinaryField()   # Stored as bytes without a dedicated vector type
    dim = fields.IntField()
    model = fields.CharField(max_length=255)

    class Meta:
        table = "embeddings"
```

```python
# app/models/tags.py
from tortoise import fields
from .base import UUIDBase

class Tag(UUIDBase):
    name = fields.CharField(max_length=64, unique=True)

    class Meta:
        table = "tags"
```

```python
# app/models/relations.py
from tortoise import fields
from .base import UUIDBase

class ChunkTag(UUIDBase):
    chunk = fields.ForeignKeyField("models.Chunk", on_delete=fields.CASCADE)
    tag = fields.ForeignKeyField("models.Tag", on_delete=fields.CASCADE)

    class Meta:
        table = "chunk_tags"
        unique_together = ("chunk", "tag")
```

### SQL & Indices (excerpt of `scripts/init_db.sql`)
```sql
CREATE EXTENSION IF NOT EXISTS vector;

-- FTS helper
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS fts tsvector
  GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;

-- Vector column (if you prefer raw SQL over Aerich migration for clarity)
ALTER TABLE embeddings
  ADD COLUMN IF NOT EXISTS vector vector(384);

-- Hybrid indices
CREATE INDEX IF NOT EXISTS idx_chunks_fts ON chunks USING GIN (fts);
CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON embeddings USING ivfflat (vector vector_l2_ops) WITH (lists = 100);

-- Dedup
CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_sha ON documents(content_sha256);
CREATE UNIQUE INDEX IF NOT EXISTS uq_chunks_sha ON chunks(chunk_sha256);
```

---

## 4) API Contracts

### `POST /ingest`
**Body**
```json
{
  "text": "string (required)",
  "tags": ["string"],
  "source": "string",
  "created_at": "iso8601"
}
```

**Validations**
- Reject if `text` is empty or `len(text) < MIN_INPUT_CHARS`.
- Reject if `content_sha256` already exists (conflict).

**Flow** (each step emits a span):
1. `normalize(text)` → strip, collapse whitespace, normalize quotes.
2. `hash(text)` → `content_sha256`.
3. `chunk(text)` → simple sentence/paragraph split with N-sentence overlap.
4. `embed(chunks)` → batched embeddings; store (per-chunk).
5. `tagging` → upsert `Tag` rows; link via `ChunkTag`.
6. Return `{document_id, chunk_count, tags[]}`.

**Responses**
- `201 Created`: `{ document_id, chunk_ids[], chunk_count, tags[] }`
- `409 Conflict`: duplicate by content hash
- `422 Unprocessable Entity`: too short / invalid

### `GET /answer?q=...&k=12&eval=true|false`
**Flow** (LangGraph):
```
parse → retrieve(hybrid, k) → rerank → compose answer (DSPy) → cite
```

**Contract (always returns citations)**
```json
{
  "answer": "string",
  "citations": [
    { "chunk_id": "uuid", "preview": "first ~160 chars...", "score": 0.76 }
  ],
  "confidence": 0.0_to_1.0,
  "debug": { "steps": "... optional when APP_ENV=dev" }
}
```

**Query params**
- `k` (default from env, capped)
- `eval` (default `false`): if `true`, compute & log hit-rate metrics (see §7).

---

## 5) RAG Implementation Details

### 5.1 Chunking (naive, simple but robust)
- Split on blank-line paragraphs.
- Further split long paragraphs into N-sentence windows with **1-sentence overlap**.
- `text_preview` := first 160 chars of chunk, sanitized.

### 5.2 Embedding
- Default to `all-MiniLM-L6-v2` (dim=384) or plug your own by env.
- Store vectors in `embeddings.vector` (pgvector).
- Batch size (e.g., 64) with a trace span for the batch.

### 5.3 Hybrid Retrieval
- **Vector**: ANN search on `embeddings.vector` by cosine/L2.
- **Lexical**: FTS query on `chunks.fts` with `plainto_tsquery`.
- **Fusion**: normalize scores (min-max), `score = α * vector + (1-α) * lexical` (α ~ 0.6 for short queries).
- Optional tag filtering: `WHERE tag IN (...)` via join on `ChunkTag`.

**Example SQL (vector leg)**
```sql
SELECT c.id, c.text_preview, 1 - (e.vector <=> :qvec) AS vscore
FROM embeddings e
JOIN chunks c ON c.id = e.chunk_id
ORDER BY e.vector <-> :qvec
LIMIT :kvec;
```

**Example SQL (lexical leg)**
```sql
SELECT id, text_preview, ts_rank(fts, plainto_tsquery('english', :q)) AS lscore
FROM chunks
WHERE fts @@ plainto_tsquery('english', :q)
ORDER BY lscore DESC
LIMIT :klex;
```

**Fusion (Python)**
```python
fused = fuse(vector_hits, lexical_hits, alpha=0.6)[:k]
```

### 5.4 Reranking
- Lightweight cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) **or** small LLM scoring.
- Input: `(query, chunk_text)`
- Output: `score ∈ [0, 1]`; keep top-k' (e.g., 6).

### 5.5 Answer Composition (DSPy + LangGraph)
- **LangGraph nodes**:
  - **parse**: normalize/expand the question.
  - **retrieve**: hybrid retrieval (k).
  - **rerank**: re-score top-k; select k'.
  - **answer**: DSPy program (below) to synthesize grounded answer.
  - **cite**: attach `{chunk_id, preview, score}` for all chunks referenced.

- **DSPy Signature (strict)**
```python
# app/rag/dspy_program.py
import dspy

class AnswerSignature(dspy.Signature):
    """Answer a question concisely using only the provided context.""" 
    question = dspy.InputField(desc="natural language query")
    context = dspy.InputField(desc="relevant passages with ids")
    answer = dspy.OutputField()
    citations = dspy.OutputField()   # list of chunk ids used
    confidence = dspy.OutputField()  # float 0..1

Answerer = dspy.Program(AnswerSignature)
```

- The **answer** node prepares a `context` string like:
  ```
  [chunk_id=... score=0.82]
  ...chunk text...

  [chunk_id=... score=0.77]
  ...chunk text...
  ```
  and calls the DSPy program. The node enforces schema and post-validates:
  - `citations` must be a subset of retrieved chunk IDs.
  - If missing, fall back to top reranked chunks.

---

## 6) API Handlers (sketch)

```python
# app/api/ingest.py
@router.post("/ingest", status_code=201)
async def ingest(payload: IngestRequest) -> IngestResponse:
    with tracer.start_as_current_span("ingest") as span:
        doc_id, chunk_ids = await ingestion_service.ingest(payload)
        meter.counter("ingest.requests").add(1)
        return {"document_id": str(doc_id), "chunk_ids": [str(i) for i in chunk_ids], "chunk_count": len(chunk_ids), "tags": payload.tags or []}
```

```python
# app/api/answer.py
@router.get("/answer")
async def answer(q: str, k: int = settings.RETRIEVAL_K, eval: bool = False):
    with tracer.start_as_current_span("answer") as span:
        result = await qa_service.answer(q=q, k=k, eval_mode=eval)
        meter.counter("answer.requests").add(1)
        meter.histogram("answer.latency_ms").record(span.end_time - span.start_time)  # pseudo
        return result
```

---

## 7) Observability (OTEL)

### Tracing
- **/ingest** spans:
  - `normalize`, `hash`, `chunking`, `embedding(batch=N)`,
  - `db.write.document`, `db.write.chunks`, `db.write.embeddings`, `db.write.tags`.
- **/answer** spans:
  - `parse`, `retrieve.vector`, `retrieve.lexical`, `retrieve.fuse`,
  - `rerank`, `answer.dspy`, `cite`,
  - `db.read.vector`, `db.read.lexical`.

Use OTEL auto-instrumentation where possible:
- FastAPI/ASGI, `asyncpg` (DB), `requests`/`httpx` (LLM calls).

### Metrics (via OTEL Meter)
- **Counters**: `ingest.requests`, `answer.requests`.
- **Histograms**: `ingest.latency_ms`, `answer.latency_ms`.
- **LLM tokens**: `llm.tokens.prompt`, `llm.tokens.completion`.
- **Retrieval** (if `eval=true`):
  - `retrieval.hit_rate_at_k` (0/1 per request; gold label optional),
  - `retrieval.mean_score_top1`, `retrieval.mean_score_topk`.

All metrics are labeled with `{env, model, route}`.

---

## 8) Local Development

### Bootstrap DB
1. `docker compose up -d` (optional) — brings Postgres with `pgvector`.
2. `aerich init-db` — creates migration history in DB.
3. `aerich migrate` — runs migrations.

### Run app
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Smoke tests
```bash
curl -X POST 'http://localhost:8000/ingest'   -H 'content-type: application/json'   -d '{"text":"Alice met Bob.\n\nThey built a PKB MVP.","tags":["demo","notes"],"source":"cli"}'

curl 'http://localhost:8000/answer?q=Who met whom?&k=8'
```

---

## 9) Key Implementation Snippets

### 9.1 Tortoise ORM bootstrap

```python
# app/core/db.py
from fastapi import FastAPI
from tortoise.contrib.fastapi import register_tortoise
from app.core.config import settings

def init_tortoise(app: FastAPI):
    register_tortoise(
        app,
        db_url=settings.DATABASE_URL,
        modules={"models": ["app.models"]},
        generate_schemas=True,
        add_exception_handlers=True,
    )
```

```python
# aerich.ini
[tortoise]
# tortoise_orm config, same as in register_tortoise
tortoise_orm = {
    "connections": {"default": "postgres://pkb:pkb@localhost:5432/pkb"},
    "apps": {
        "models": {
            "models": ["app.models", "aerich.models"],
            "default_connection": "default",
        },
    },
}
```

### 9.2 Naive chunker

```python
# app/rag/chunker.py
def split(text: str, max_sent=6, overlap=1):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    out = []
    for p in paras:
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', p) if s.strip()]
        for i in range(0, len(sents), max_sent - overlap):
            window = sents[i:i+max_sent]
            if window:
                out.append(" ".join(window))
    return out
```

### 9.3 Hybrid retriever (sketch)

```python
# app/rag/retriever.py
async def hybrid(query: str, k: int) -> list[Hit]:
    vhits = await vector_search(query_vec, kvec)
    lhits = await lexical_search(query, klex)
    return fuse(vhits, lhits, alpha=0.6)[:k]
```

### 9.4 LangGraph wiring (sketch)

```python
# app/rag/graph.py
from langgraph.graph import StateGraph

def build_graph():
    g = StateGraph(dict)
    g.add_node("parse", parse_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("rerank", rerank_node)
    g.add_node("answer", answer_node)
    g.add_node("cite", cite_node)
    g.set_entry_point("parse")
    g.add_edge("parse", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "answer")
    g.add_edge("answer", "cite")
    return g.compile()
```

### 9.5 DSPy program contract enforcement

```python
# app/rag/dspy_program.py
def run_answerer(question: str, passages: list[tuple[str, str, float]]):
    context = "\n\n".join([f"[chunk_id={cid} score={s:.2f}]\n{txt}" for cid, txt, s in passages])
    out = Answerer(question=question, context=context)
    # Post-validate
    out.citations = [c for c in out.citations if c in {cid for cid,_,_ in passages}]
    out.confidence = max(0.0, min(1.0, float(out.confidence)))
    return {"answer": out.answer, "citations": out.citations, "confidence": out.confidence}
```

---

## 10) Error Handling & Edge Cases

- Ingestion:
  - 409 on duplicate `content_sha256`.
  - 422 if too short or non-text payload.
  - Partial failure on embeddings: store what’s computed, return count; log errors.

- Answering:
  - If retrieval yields 0, return `"answer": ""`, `citations: []`, `confidence: 0.0` with 200 OK.
  - Guardrail: cap `k` to `<= 50`.

---

## 11) Security (MVP)

- Limit payload sizes (e.g., 200k chars).
- Strip control characters; reject binary.
- Parameterized SQL only (asyncpg).
- No auth (local dev), but structure allows API-key header later.

---

## 12) “How it all flows” (text diagram)

```
POST /ingest
  ↓
normalize ── hash ── dedup? ── chunk ── embed(batch) ── store(doc, chunks, embeddings, tags)
  ↓
  201 {document_id, chunk_count, tags}

GET /answer?q=...
  ↓
LangGraph: parse → retrieve(hybrid k) → rerank → DSPy answer → cite
  ↓
  200 {answer, citations[{chunk_id, preview, score}], confidence}
```

---

## 13) Optional: docker-compose (local only)

```yaml
version: "3.9"
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: pkb
      POSTGRES_USER: pkb
      POSTGRES_PASSWORD: pkb
    ports: ["5432:5432"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U pkb -d pkb"]
      interval: 5s
      timeout: 5s
      retries: 20
```

---

## 14) Minimal Test Plan

- Ingest small doc → assert 201, chunk_count > 0, embeddings created.
- Ingest same doc → assert 409.
- Answer with query that exists → citations non-empty; confidence in [0,1].
- Answer with nonsense → citations empty; confidence low.
- `eval=true` → hit-rate metric emitted to OTEL meter.

---

## 15) Future-proofing (kept out of MVP)

- Auth & multi-tenant.
- Advanced chunking (semantic breakpoints).
- Learned fusion weights & query rewriting.
- Background workers (Celery) for large ingests.
- UI for search/inspect.
