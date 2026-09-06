"""Integration tests for document ingestion API endpoints."""

import io

from fastapi.testclient import TestClient


def test_ingest_text_endpoint(test_client: TestClient) -> None:
    """Verify POST /api/v1/documents/ingest/text ingests and chunks content."""
    payload = {
        "title": "Data Retention Standard",
        "content": (
            "# Data Retention Policy\n\n"
            "All customer communication logs must be archived for 7 years.\n\n"
            "Payment card details must not be stored in plain text."
        ),
        "source": "governance_portal",
    }

    response = test_client.post("/api/v1/documents/ingest/text", json=payload)
    assert response.status_code == 201

    data = response.json()
    assert data["status"] == "success"
    assert data["title"] == "Data Retention Standard"
    assert data["num_chunks"] > 0
    assert len(data["checksum"]) == 64
    assert "document_id" in data


def test_ingest_duplicate_text_skips(test_client: TestClient) -> None:
    """Verify identical text ingestion returns skipped_duplicate status."""
    content = "Single unique sentence for idempotency testing."
    payload = {
        "title": "Unique Document",
        "content": content,
    }

    # First ingestion
    res1 = test_client.post("/api/v1/documents/ingest/text", json=payload)
    assert res1.status_code == 201
    assert res1.json()["status"] == "success"

    # Second ingestion
    res2 = test_client.post("/api/v1/documents/ingest/text", json=payload)
    assert res2.status_code == 201
    assert res2.json()["status"] == "skipped_duplicate"
    assert res2.json()["document_id"] == res1.json()["document_id"]


def test_ingest_file_endpoint(test_client: TestClient) -> None:
    """Verify POST /api/v1/documents/ingest/file supports multipart file upload."""
    file_content = (
        b"# Incident Response Plan\n\nP1 incidents must be acknowledged within 15 minutes."
    )
    file_tuple = ("incident_response.md", io.BytesIO(file_content), "text/markdown")

    response = test_client.post(
        "/api/v1/documents/ingest/file",
        files={"file": file_tuple},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "success"
    assert data["title"] == "Incident Response Plan"
    assert data["num_chunks"] > 0


def test_ingest_empty_file_rejected(test_client: TestClient) -> None:
    """Verify empty file upload returns 400 Bad Request."""
    empty_tuple = ("empty.txt", io.BytesIO(b""), "text/plain")
    response = test_client.post(
        "/api/v1/documents/ingest/file",
        files={"file": empty_tuple},
    )
    assert response.status_code == 400


def test_list_and_get_chunks_endpoints(test_client: TestClient) -> None:
    """Verify GET /documents and GET /documents/{id}/chunks."""
    # 1. Ingest a document to inspect
    payload = {
        "title": "Vacation SOP",
        "content": "All employees receive 20 paid vacation days and 5 sick days annually.",
        "source": "hr_portal",
    }
    ingest_res = test_client.post("/api/v1/documents/ingest/text", json=payload)
    assert ingest_res.status_code == 201
    doc_id = ingest_res.json()["document_id"]

    # 2. List documents
    list_res = test_client.get("/api/v1/documents")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["total_documents"] >= 1
    matching = [d for d in list_data["documents"] if d["id"] == doc_id]
    assert len(matching) == 1

    # 3. Retrieve chunks for document
    chunks_res = test_client.get(f"/api/v1/documents/{doc_id}/chunks")
    assert chunks_res.status_code == 200
    chunks = chunks_res.json()
    assert len(chunks) > 0
    assert chunks[0]["document_id"] == doc_id
    assert "Vacation" in chunks[0]["metadata"]["title"]

    # 4. Non-existent document 404
    missing_res = test_client.get("/api/v1/documents/non-existent-id/chunks")
    assert missing_res.status_code == 404
