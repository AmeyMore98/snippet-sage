# Async Ingestion — Implementation Plan

> This plan details the granular tasks required to make the `POST /ingest` endpoint asynchronous, as scoped in `async_ingest.md`.

---

## Phase 0 — Dependencies & Configuration - Done

### T0.1 — Add Dependencies

**Start:** `requirements.txt` is missing job queue libraries.
**End:** `dramatiq` and `redis` are added to `requirements.txt`.
**Steps:**

1.  Append `dramatiq` and `redis` to `requirements.txt`.
    **Test:** `pip install -r requirements.txt` completes successfully.

### T0.2 — Add Redis to Docker Compose

**Start:** `docker-compose.yml` only contains the database.
**End:** A Redis service is added to `docker-compose.yml`.
**Steps:**

1.  Add a `redis` service using the official `redis:alpine` image.
2.  Expose port `6379`.
    **Test:** `docker compose up -d redis` starts the container successfully.

### T0.3 — Update Configuration

**Start:** `app/core/config.py` lacks Redis settings.
**End:** `Settings` model includes `REDIS_URL`.
**Steps:**

1.  Add `REDIS_URL` to `.env.example` (e.g., `redis://localhost:6379/0`).
2.  Add `REDIS_URL` to the `Settings` class in `app/core/config.py`.
    **Test:** `python -c "from app.core.config import Settings; print(Settings().REDIS_URL)"` prints the correct URL.

---

## Phase 1 — Queue & Worker Setup - Done

### T1.1 — Implement Dramatiq Broker

**Start:** No queueing mechanism exists.
**End:** A Dramatiq broker is configured in `app/core/queue.py`.
**Steps:**

1.  Create `app/core/queue.py`.
2.  Import `dramatiq`, `RedisBroker`, and the `settings` object.
3.  Initialize and configure a `RedisBroker` instance.
4.  Set the broker as the global Dramatiq broker.
    **Test:** The module imports without error.

### T1.2 — Add Worker Startup Command

**Start:** No way to run the job worker.
**End:** Documentation or a script is available to start the worker.
**Steps:**

1.  Add a `run_worker.sh` script or update `README.md` with the command: `dramatiq app.jobs.ingestion_job`.
    **Test:** The command is documented and clear.

---

## Phase 2 — Data Model Changes - Done

### T2.1 — Update Document Model

**Start:** `Document` model has no status tracking.
**End:** `Document` model in `app/models/document.py` has `status` and `processing_errors` fields.
**Steps:**

1.  Add a `status` field (e.g., `fields.CharField`) with an enum of choices: `PENDING`, `PROCESSING`, `COMPLETED`, `FAILED`. Set `default="PENDING"`.
2.  Add a `processing_errors` field (`fields.TextField(null=True)`) to store error messages.
    **Test:** `python -c "from app.models.document import Document; print(hasattr(Document, 'status'))"` prints `True`.

### T2.2 — Create and Apply Migration

**Start:** Database schema is out of sync with the model.
**End:** The new fields are added to the `documents` table in the database.
**Steps:**

1.  Run `aerich migrate` to generate a new migration file.
2.  Inspect the generated migration for correctness.
3.  Run `aerich upgrade` to apply it.
    **Test:** `\d+ documents` in `psql` shows the `status` and `processing_errors` columns.

---

## Phase 3 — Ingestion Job

### T3.1 — Create Ingestion Actor

**Start:** No job processor exists.
**End:** `app/jobs/ingestion_job.py` contains the `process_ingestion` actor.
**Steps:**

1.  Create the file `app/jobs/ingestion_job.py`.
2.  Import `dramatiq`, `IngestionService`, and the `Document` model.
3.  Define a Dramatiq actor `process_ingestion(document_id: str)` with `max_retries=5`.
    **Test:** The file is created and the actor is defined.

### T3.2 — Implement Job Logic - Done

**Start:** The actor has an empty body.
**End:** The actor orchestrates the ingestion and updates the document status.
**Steps:**

1.  Fetch the document by `document_id`.
2.  Wrap the core logic in a `try...except` block.
3.  Set status to `PROCESSING`.
4.  Instantiate `IngestionService` and call its `ingest_document` method (this method will need to be refactored from the existing `ingest` service method to operate on an existing document).
5.  On success, set status to `COMPLETED`.
    **Test:** A unit test for the job shows it calls the service and updates status correctly on success.

### T3.3 — Implement Failure Handling

**Start:** The job fails silently or without status updates.
**End:** The actor handles exceptions, logs them, and updates the status to `FAILED` after exhausting retries.
**Steps:**

1.  In the `except` block, catch `Exception as e`.
2.  Log the exception.
3.  Check if the retry limit is reached.
4.  If retries are exhausted, update the document status to `FAILED` and store the error message in `processing_errors`.
5.  Re-raise the exception to let Dramatiq handle the retry mechanism.
    **Test:** A unit test where the service mock throws an exception shows the status is updated to `FAILED` after the configured number of retries.

---

## Phase 4 — API Modification - Done

### T4.1 — Update API Schemas

**Start:** The `/ingest` response schema is designed for synchronous processing.
**End:** The response schema in `app/schemas/ingest.py` is updated for async operation.
**Steps:**

1.  Modify `IngestResponse` to only return `document_id` and a status message.
    **Test:** The schema correctly reflects the new, minimal response.

### T4.2 — Modify Ingestion Endpoint

**Start:** `POST /ingest` is a long-running, synchronous operation.
**End:** The endpoint in `app/api/ingest.py` creates a `Document` record and enqueues a job.
**Steps:**

1.  The handler will now perform only the initial steps: validation, hashing, and duplicate checking.
2.  If it is a new document, create a `Document` instance with `status="PENDING"`.
3.  Enqueue the `process_ingestion` job with the new document's ID.
4.  Return a `201 Created` response with the document ID.
    **Test:** An API test confirms the endpoint returns `201` immediately, and a mock of the Dramatiq broker shows that the `process_ingestion.send()` method was called with the correct `document_id`.

---

## Phase 5 — Testing - Done

### T5.1 — API Endpoint Unit Test

**Start:** No test for the new async behavior.
**End:** A test verifies the API handler creates a document and enqueues a job.
**Steps:**

1.  Use a test client to `POST` to `/ingest`.
2.  Assert a `201` response.
3.  Assert a `Document` was created in the test DB with `status="PENDING"`.
4.  Assert the `send()` method on a mocked Dramatiq actor was called.
    **Test:** `pytest tests/test_api.py` passes.

### T5.2 — Ingestion Job Unit Test

**Start:** No test for the actor.
**End:** A test verifies the job's success and failure logic.
**Steps:**

1.  Create a test that calls the `process_ingestion` actor directly.
2.  **Success case:** Seed a `Document` with `status="PENDING"`. Run the job and assert the status becomes `COMPLETED` and chunks/embeddings are created.
3.  **Failure case:** Mock the `IngestionService` to raise an exception. Run the job and assert the document status becomes `FAILED` and `processing_errors` is populated.
    **Test:** `pytest tests/test_jobs.py` (a new test file) passes.
