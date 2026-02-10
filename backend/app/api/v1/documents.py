import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.database import get_async_session
from app.core.exceptions import DocumentNotFoundError, FileTooLargeError, UnsupportedFileTypeError
from app.core.logging import get_logger
from app.schemas.common import SuccessResponse
from app.schemas.document import (
    BatchUploadResponse,
    DocumentChunkResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentStatus,
    DocumentUploadResponse,
)
from app.services.document_service import DocumentService
from app.services.storage_service import StorageService

router = APIRouter(prefix="/documents", tags=["Documents"])
logger = get_logger(__name__)

ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/csv",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


def _validate_file(file: UploadFile) -> None:
    # Check MIME type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise UnsupportedFileTypeError(file.content_type or "unknown")

    # Check file size (approximate from headers)
    if file.size and file.size > settings.max_file_size_mb * 1024 * 1024:
        raise FileTooLargeError(
            size_mb=file.size / (1024 * 1024),
            max_mb=settings.max_file_size_mb,
        )


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_async_session),
):
    _validate_file(file)

    storage = StorageService()
    doc_service = DocumentService(db)

    # Read file content
    content = await file.read()

    # Validate actual size
    actual_size = len(content)
    if actual_size > settings.max_file_size_mb * 1024 * 1024:
        raise FileTooLargeError(
            size_mb=actual_size / (1024 * 1024),
            max_mb=settings.max_file_size_mb,
        )

    # Generate storage key
    file_ext = file.filename.rsplit(".", 1)[-1] if "." in file.filename else ""
    storage_key = f"{uuid.uuid4()}.{file_ext}" if file_ext else str(uuid.uuid4())

    # Upload to MinIO
    storage.upload_file(
        bucket=settings.minio_bucket_raw,
        key=storage_key,
        data=content,
        content_type=file.content_type or "application/octet-stream",
    )

    # Create database record
    document = await doc_service.create_document(
        filename=storage_key,
        original_filename=file.filename or "unnamed",
        mime_type=file.content_type or "application/octet-stream",
        file_size_bytes=actual_size,
        storage_key=storage_key,
    )
    await db.commit()

    # Dispatch Celery processing task
    from app.tasks.document_tasks import process_document

    process_document.delay(str(document.id))

    return DocumentUploadResponse(
        id=document.id,
        filename=file.filename or "unnamed",
        status="pending",
    )


@router.post("/upload/batch", response_model=BatchUploadResponse)
async def upload_batch(
    files: list[UploadFile] = File(...),
    db: AsyncSession = Depends(get_async_session),
):
    results = []
    errors = []
    storage = StorageService()
    doc_service = DocumentService(db)

    for file in files:
        filename = file.filename or "unnamed"
        try:
            _validate_file(file)
            content = await file.read()
            actual_size = len(content)

            if actual_size > settings.max_file_size_mb * 1024 * 1024:
                errors.append(
                    {
                        "filename": filename,
                        "error": f"File exceeds max size of {settings.max_file_size_mb}MB",
                    }
                )
                continue

            file_ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
            storage_key = f"{uuid.uuid4()}.{file_ext}" if file_ext else str(uuid.uuid4())

            storage.upload_file(
                bucket=settings.minio_bucket_raw,
                key=storage_key,
                data=content,
                content_type=file.content_type or "application/octet-stream",
            )

            document = await doc_service.create_document(
                filename=storage_key,
                original_filename=filename,
                mime_type=file.content_type or "application/octet-stream",
                file_size_bytes=actual_size,
                storage_key=storage_key,
            )

            results.append(
                DocumentUploadResponse(
                    id=document.id,
                    filename=filename,
                    status="pending",
                )
            )
        except (UnsupportedFileTypeError, FileTooLargeError) as e:
            errors.append({"filename": filename, "error": str(e)})
        except Exception as e:
            logger.warning("Batch upload failed for %s: %s", filename, e)
            errors.append({"filename": filename, "error": "Upload failed"})
            continue

    await db.commit()

    # Dispatch processing tasks
    from app.tasks.document_tasks import process_document

    for result in results:
        process_document.delay(str(result.id))

    return BatchUploadResponse(
        documents=results,
        total_uploaded=len(results),
        errors=errors,
    )


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
    status: Optional[str] = Query(default=None),
    mime_type: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_async_session),
):
    doc_service = DocumentService(db)
    documents, total = await doc_service.list_documents(
        page=page, size=size, status=status, mime_type=mime_type
    )

    pages = (total + size - 1) // size
    return DocumentListResponse(
        total=total,
        page=page,
        size=size,
        pages=pages,
        items=[DocumentResponse.model_validate(doc) for doc in documents],
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
):
    doc_service = DocumentService(db)
    document = await doc_service.get_document(document_id)
    if not document:
        raise DocumentNotFoundError(str(document_id))
    return DocumentResponse.model_validate(document)


@router.get("/{document_id}/status", response_model=DocumentStatus)
async def get_document_status(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
):
    doc_service = DocumentService(db)
    document = await doc_service.get_document(document_id)
    if not document:
        raise DocumentNotFoundError(str(document_id))
    return DocumentStatus(
        id=document.id,
        status=document.status,
        error_message=document.error_message,
        page_count=document.page_count,
        total_chunks=document.total_chunks,
    )


@router.get("/{document_id}/chunks", response_model=list[DocumentChunkResponse])
async def get_document_chunks(
    document_id: uuid.UUID,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_async_session),
):
    doc_service = DocumentService(db)
    document = await doc_service.get_document(document_id)
    if not document:
        raise DocumentNotFoundError(str(document_id))

    chunks = await doc_service.get_chunks(document_id, page=page, size=size)
    return [DocumentChunkResponse.model_validate(c) for c in chunks]


@router.delete("/{document_id}", response_model=SuccessResponse)
async def delete_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
):
    doc_service = DocumentService(db)
    document = await doc_service.get_document(document_id)
    if not document:
        raise DocumentNotFoundError(str(document_id))

    # Delete from MinIO
    storage = StorageService()
    try:
        storage.delete_file(settings.minio_bucket_raw, document.storage_key)
    except Exception:
        pass

    await doc_service.delete_document(document_id)
    await db.commit()
    return SuccessResponse(message=f"Document {document_id} deleted")
