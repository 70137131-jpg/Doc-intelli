from pydantic import BaseModel, Field


class PaginationParams(BaseModel):
    page: int = Field(default=1, ge=1)
    size: int = Field(default=20, ge=1, le=100)

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.size


class PaginatedResponse(BaseModel):
    total: int
    page: int
    size: int
    pages: int
    items: list


class ErrorResponse(BaseModel):
    detail: str
    status_code: int = 500


class SuccessResponse(BaseModel):
    message: str
    success: bool = True
