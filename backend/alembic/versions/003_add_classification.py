"""Add classification and extracted_fields tables

Revision ID: 003
Revises: 002
Create Date: 2026-02-10

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Classifications table
    op.create_table(
        "classifications",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("document_type", sa.String(100), nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("method", sa.String(50), nullable=False),
        sa.Column("raw_scores", JSONB, nullable=True),
        sa.Column("reasoning", sa.Text, nullable=True),
        sa.Column("is_corrected", sa.Boolean, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_classifications_document_type", "classifications", ["document_type"])

    # Extracted fields table
    op.create_table(
        "extracted_fields",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("field_name", sa.String(200), nullable=False),
        sa.Column("field_value", sa.Text, nullable=True),
        sa.Column("field_type", sa.String(50), nullable=True),
        sa.Column("confidence", sa.Float, nullable=True),
        sa.Column("extraction_method", sa.String(50), nullable=True),
        sa.Column("is_corrected", sa.Boolean, server_default="false"),
        sa.Column("original_value", sa.Text, nullable=True),
        sa.Column("metadata", JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_extracted_fields_document_id", "extracted_fields", ["document_id"])
    op.create_index("idx_extracted_fields_field_name", "extracted_fields", ["field_name"])
    op.create_index("idx_extracted_fields_unique", "extracted_fields", ["document_id", "field_name"], unique=True)


def downgrade() -> None:
    op.drop_table("extracted_fields")
    op.drop_table("classifications")
