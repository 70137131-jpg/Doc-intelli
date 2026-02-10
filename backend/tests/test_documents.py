import pytest
from httpx import AsyncClient


class TestHealthEndpoints:
    @pytest.mark.asyncio
    async def test_health_check(self, client: AsyncClient):
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_readiness_check(self, client: AsyncClient):
        response = await client.get("/api/v1/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert "checks" in data


class TestDocumentEndpoints:
    @pytest.mark.asyncio
    async def test_list_documents_empty(self, client: AsyncClient):
        response = await client.get("/api/v1/documents/")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["items"] == []

    @pytest.mark.asyncio
    async def test_get_nonexistent_document(self, client: AsyncClient):
        response = await client.get("/api/v1/documents/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_upload_unsupported_file(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.exe", b"fake content", "application/x-msdownload")},
        )
        assert response.status_code == 400
