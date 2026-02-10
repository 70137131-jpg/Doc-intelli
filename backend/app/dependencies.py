from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_session
from app.services.document_service import DocumentService
from app.services.storage_service import StorageService


async def get_document_service(
    db: AsyncSession = Depends(get_async_session),
) -> DocumentService:
    return DocumentService(db)


def get_storage_service() -> StorageService:
    return StorageService()
