"""Load testing with Locust for Doc Intelli API.

Run:
    locust -f tests/load/locustfile.py --host http://localhost:8000
    Then open http://localhost:8089 to configure and start the test.
"""

import json
import random
import uuid

from locust import HttpUser, between, task


class DocIntelliUser(HttpUser):
    """Simulates a typical Doc Intelli user workflow."""

    wait_time = between(1, 3)
    token: str | None = None
    document_ids: list[str] = []

    def on_start(self):
        """Register and login to get JWT token."""
        email = f"loadtest-{uuid.uuid4().hex[:8]}@test.com"
        password = "LoadTest123!"

        # Register
        self.client.post(
            "/api/v1/auth/register",
            json={
                "email": email,
                "password": password,
                "full_name": "Load Test User",
            },
        )

        # Login
        resp = self.client.post(
            "/api/v1/auth/login",
            json={"email": email, "password": password},
        )
        if resp.status_code == 200:
            data = resp.json()
            self.token = data.get("access_token")

    @property
    def auth_headers(self):
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}

    @task(3)
    def list_documents(self):
        """List documents — most common operation."""
        resp = self.client.get(
            "/api/v1/documents/?limit=20",
            headers=self.auth_headers,
            name="/api/v1/documents/",
        )
        if resp.status_code == 200:
            data = resp.json()
            docs = data.get("documents", [])
            self.document_ids = [d["id"] for d in docs]

    @task(2)
    def search_documents(self):
        """Hybrid search query."""
        queries = [
            "invoice total amount",
            "contract termination clause",
            "quarterly revenue report",
            "employee benefits summary",
            "project timeline milestones",
        ]
        self.client.post(
            "/api/v1/search/",
            json={
                "query": random.choice(queries),
                "mode": "hybrid",
                "top_k": 10,
            },
            headers=self.auth_headers,
            name="/api/v1/search/",
        )

    @task(2)
    def chat_query(self):
        """Send a chat/RAG query."""
        questions = [
            "What is the total amount on the latest invoice?",
            "Summarize the contract terms.",
            "What are the key findings in the report?",
            "List the action items from the meeting notes.",
            "What is the revenue for Q3?",
        ]
        self.client.post(
            "/api/v1/chat/query",
            json={"query": random.choice(questions)},
            headers=self.auth_headers,
            name="/api/v1/chat/query",
        )

    @task(1)
    def get_document_detail(self):
        """View a specific document's details."""
        if not self.document_ids:
            return
        doc_id = random.choice(self.document_ids)
        self.client.get(
            f"/api/v1/documents/{doc_id}",
            headers=self.auth_headers,
            name="/api/v1/documents/[id]",
        )

    @task(1)
    def get_classifications(self):
        """List classifications."""
        self.client.get(
            "/api/v1/classification/",
            headers=self.auth_headers,
            name="/api/v1/classification/",
        )

    @task(1)
    def upload_document(self):
        """Upload a small test document."""
        content = f"This is a load test document {uuid.uuid4().hex}.\n" * 10
        self.client.post(
            "/api/v1/documents/upload",
            files={"file": ("loadtest.txt", content.encode(), "text/plain")},
            headers=self.auth_headers,
            name="/api/v1/documents/upload",
        )

    @task(1)
    def health_check(self):
        """Health check endpoint."""
        self.client.get("/health", name="/health")

    @task(1)
    def get_me(self):
        """Get current user profile."""
        self.client.get(
            "/api/v1/auth/me",
            headers=self.auth_headers,
            name="/api/v1/auth/me",
        )
