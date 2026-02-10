"""Add pgvector extension and chunk_embeddings table

Revision ID: 002
Revises: 001
Create Date: 2026-02-10

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create chunk_embeddings table using raw SQL for vector column support
    op.execute("""
        CREATE TABLE chunk_embeddings (
            id UUID PRIMARY KEY,
            chunk_id UUID NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
            document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            embedding vector(384) NOT NULL,
            model_name VARCHAR(200) NOT NULL DEFAULT 'all-MiniLM-L6-v2',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        )
    """)

    op.execute("""
        CREATE INDEX idx_chunk_embeddings_hnsw ON chunk_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

    op.execute("""
        CREATE INDEX idx_chunk_embeddings_document_id ON chunk_embeddings(document_id)
    """)

    op.execute("""
        CREATE UNIQUE INDEX idx_chunk_embeddings_unique ON chunk_embeddings(chunk_id, model_name)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS chunk_embeddings")
    op.execute("DROP EXTENSION IF EXISTS vector")
