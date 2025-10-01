from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.schemas.answer import AnswerResponse, Citation


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.anyio
class TestAPI:
    async def test_health_check(self, client: AsyncClient):
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @patch("app.jobs.ingestion_job.process_ingestion")
    async def test_ingest_endpoint_creates_document_and_enqueues_job(self, mock_actor, client: AsyncClient):
        """Tests that the async ingest endpoint creates a document and enqueues a job."""
        test_text = "This is a test document for async ingestion."
        payload = {"text": test_text, "source": "api-test"}

        response = await client.post("/api/v1/ingest", json=payload)
        assert response.status_code == 201
        data = response.json()

        # Verify response contains document_id and status message
        assert "document_id" in data
        assert "message" in data
        assert data["message"] == "Document accepted for processing"

        # Verify the Dramatiq actor was called with the document ID
        mock_actor.send.assert_called_once()
        call_args = mock_actor.send.call_args[0]
        assert call_args[0] == data["document_id"]

    @patch("app.jobs.ingestion_job.process_ingestion")
    async def test_ingest_endpoint_duplicate_returns_same_id(self, mock_actor, client: AsyncClient):
        """Tests that ingesting duplicate content returns the same document and enqueues a new job."""
        test_text = "This is a duplicate test document."
        payload = {"text": test_text, "source": "api-test"}

        # First request
        response1 = await client.post("/api/v1/ingest", json=payload)
        assert response1.status_code == 201
        data1 = response1.json()
        assert "document_id" in data1
        assert data1["message"] == "Document accepted for processing"

        # Second (duplicate) request
        mock_actor.reset_mock()
        response2 = await client.post("/api/v1/ingest", json=payload)
        assert response2.status_code == 201
        data2 = response2.json()

        # Verify that the returned document ID is the same
        assert data1["document_id"] == data2["document_id"]
        assert data2["message"] == "Document already exists"

        # Verify that job was enqueued for the duplicate
        mock_actor.send.assert_called_once()

    @patch("app.api.answer.qa_service", new_callable=AsyncMock)
    async def test_answer_endpoint(self, mock_qa_service, client: AsyncClient):
        mock_qa_service.answer.return_value = AnswerResponse(
            answer="This is a mock answer.",
            citations=[
                Citation(
                    chunk_id="a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
                    preview="mock preview",
                    score=0.9,
                )
            ],
            confidence=0.95,
        )

        response = await client.post("/api/v1/answer", json={"q": "What does the fox do?"})
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "This is a mock answer."
        assert len(data["citations"]) == 1
        assert data["confidence"] == 0.95

    async def test_answer_empty_query(self, client: AsyncClient):
        response = await client.post("/api/v1/answer", json={"q": ""})
        assert response.status_code == 422
