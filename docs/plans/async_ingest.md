# Async Ingestion

## Scope

Make the [ingestion API](../../app//api/ingest.py) asynchronous. It should receive the payload, create an async job and return 201 to the user. The job should process the document and create chunks and embeddings.

## Design

The API handler will perform validations on payload. Once validated it checks if the document is duplicate. If not duplicate, then insert the details in `Document` table(refer this [model](../../app/models/document.py)), with status as `ingested`. We'll need a new `status` column in this table.

Create a job in a Dramatiq + Redis queue, with the document ID, ingestion timestamp and other details.

The job processor performs the actual ingestion using [IngestionService](../../app/services/ingestion_service.py). If successful will update the status to `processed` and finish the job. In case of failure do 5 retries. If retries are exhausted, updated `status` to `failed` and finish the job.

Everything else remains as is.
