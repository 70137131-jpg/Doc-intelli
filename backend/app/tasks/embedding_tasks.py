import uuid

from app.tasks.celery_app import celery_app
from app.core.database import SyncSessionLocal
from app.core.logging import get_logger
from app.models.document import Document, DocumentChunk
from app.models.embedding import ChunkEmbedding

logger = get_logger(__name__)

# Lazy-loaded model singleton
_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        from app.config import settings

        _embedding_model = SentenceTransformer(settings.embedding_model_name)
        logger.info(f"Loaded embedding model: {settings.embedding_model_name}")
    return _embedding_model


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def generate_embeddings_task(self, document_id: str):
    """Generate embeddings for all chunks of a document."""
    doc_uuid = uuid.UUID(document_id)
    session = SyncSessionLocal()

    try:
        document = session.query(Document).filter(Document.id == doc_uuid).first()
        if not document:
            logger.error(f"Document {document_id} not found")
            return

        # Get all chunks
        chunks = (
            session.query(DocumentChunk)
            .filter(DocumentChunk.document_id == doc_uuid)
            .order_by(DocumentChunk.chunk_index)
            .all()
        )

        if not chunks:
            logger.warning(f"No chunks found for document {document_id}")
            return

        # Load model
        model = get_embedding_model()

        # Generate embeddings in batches
        batch_size = 64
        texts = [chunk.content for chunk in chunks]
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = model.encode(batch, show_progress_bar=False)
            all_embeddings.extend(embeddings)

        # Store embeddings
        for chunk, embedding in zip(chunks, all_embeddings):
            # Check if embedding already exists
            existing = (
                session.query(ChunkEmbedding)
                .filter(ChunkEmbedding.chunk_id == chunk.id)
                .first()
            )

            if existing:
                existing.embedding = embedding.tolist()
            else:
                ce = ChunkEmbedding(
                    chunk_id=chunk.id,
                    document_id=doc_uuid,
                    embedding=embedding.tolist(),
                    model_name="all-MiniLM-L6-v2",
                )
                session.add(ce)

        session.commit()
        logger.info(
            f"Generated {len(all_embeddings)} embeddings for document {document_id}"
        )

    except Exception as exc:
        session.rollback()
        logger.error(
            f"Embedding generation failed for {document_id}: {exc}", exc_info=True
        )
        raise self.retry(exc=exc)
    finally:
        session.close()
