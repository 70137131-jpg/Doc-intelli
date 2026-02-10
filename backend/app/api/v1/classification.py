import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_session
from app.core.exceptions import DocumentNotFoundError
from app.schemas.classification import (
    BatchClassifyRequest,
    ClassificationResponse,
    ExtractedFieldsResponse,
    ExtractedFieldResponse,
    FieldCorrectionRequest,
)
from app.schemas.common import SuccessResponse
from app.services.document_service import DocumentService
from app.services.classification_service import ClassificationService
from app.services.field_extraction_service import FieldExtractionService

router = APIRouter(prefix="/documents", tags=["Classification"])


@router.get("/{document_id}/classification", response_model=ClassificationResponse)
async def get_classification(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
):
    doc_service = DocumentService(db)
    document = await doc_service.get_document(document_id)
    if not document:
        raise DocumentNotFoundError(str(document_id))

    cls_service = ClassificationService(db)
    classification = await cls_service.get_classification(document_id)
    if not classification:
        raise DocumentNotFoundError(f"No classification found for document {document_id}")

    return ClassificationResponse.model_validate(classification)


@router.post("/{document_id}/classify", response_model=ClassificationResponse)
async def classify_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
):
    doc_service = DocumentService(db)
    document = await doc_service.get_document(document_id)
    if not document:
        raise DocumentNotFoundError(str(document_id))

    cls_service = ClassificationService(db)
    classification = await cls_service.classify_document(document)
    await db.commit()
    return ClassificationResponse.model_validate(classification)


@router.get("/{document_id}/extracted-fields", response_model=ExtractedFieldsResponse)
async def get_extracted_fields(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
):
    doc_service = DocumentService(db)
    document = await doc_service.get_document(document_id)
    if not document:
        raise DocumentNotFoundError(str(document_id))

    field_service = FieldExtractionService(db)
    fields = await field_service.get_fields(document_id)

    cls_service = ClassificationService(db)
    classification = await cls_service.get_classification(document_id)
    doc_type = classification.document_type if classification else None

    return ExtractedFieldsResponse(
        document_id=document_id,
        document_type=doc_type,
        fields=[ExtractedFieldResponse.model_validate(f) for f in fields],
    )


@router.patch("/{document_id}/fields", response_model=SuccessResponse)
async def correct_fields(
    document_id: uuid.UUID,
    request: FieldCorrectionRequest,
    db: AsyncSession = Depends(get_async_session),
):
    doc_service = DocumentService(db)
    document = await doc_service.get_document(document_id)
    if not document:
        raise DocumentNotFoundError(str(document_id))

    field_service = FieldExtractionService(db)
    await field_service.apply_corrections(document_id, request.corrections)
    await db.commit()
    return SuccessResponse(message=f"Corrected {len(request.corrections)} fields")


@router.post("/batch/classify", response_model=SuccessResponse)
async def batch_classify(
    request: BatchClassifyRequest,
    db: AsyncSession = Depends(get_async_session),
):
    from app.tasks.classification_tasks import classify_document_task

    for doc_id in request.document_ids:
        classify_document_task.delay(str(doc_id))

    return SuccessResponse(
        message=f"Classification triggered for {len(request.document_ids)} documents"
    )
