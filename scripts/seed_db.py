"""Seed the database with sample documents for development/demo."""

import os
import sys
import uuid

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.base import Base
from app.models.document import Document


def seed():
    database_url = os.getenv(
        "SYNC_DATABASE_URL",
        "postgresql://docintelli:docintelli_dev@localhost:5432/docintelli",
    )
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Check if documents already exist
    count = session.query(Document).count()
    if count > 0:
        print(f"Database already has {count} documents. Skipping seed.")
        session.close()
        return

    print("Database is empty. Upload documents via the API to get started.")
    print("  POST http://localhost:8000/api/v1/documents/upload")
    print("  Use multipart/form-data with a 'file' field")

    session.close()


if __name__ == "__main__":
    seed()
