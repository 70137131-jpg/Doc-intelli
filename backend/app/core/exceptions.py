class DocIntelliError(Exception):
    """Base exception for the application."""

    def __init__(self, message: str = "An error occurred", status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class DocumentNotFoundError(DocIntelliError):
    def __init__(self, document_id: str):
        super().__init__(f"Document {document_id} not found", status_code=404)


class DocumentProcessingError(DocIntelliError):
    def __init__(self, message: str = "Document processing failed"):
        super().__init__(message, status_code=500)


class UnsupportedFileTypeError(DocIntelliError):
    def __init__(self, mime_type: str):
        super().__init__(f"Unsupported file type: {mime_type}", status_code=400)


class FileTooLargeError(DocIntelliError):
    def __init__(self, size_mb: float, max_mb: int):
        super().__init__(
            f"File size {size_mb:.1f}MB exceeds maximum {max_mb}MB",
            status_code=413,
        )


class StorageError(DocIntelliError):
    def __init__(self, message: str = "Storage operation failed"):
        super().__init__(message, status_code=500)


class LLMError(DocIntelliError):
    def __init__(self, message: str = "LLM API call failed"):
        super().__init__(message, status_code=502)


class ClassificationError(DocIntelliError):
    def __init__(self, message: str = "Classification failed"):
        super().__init__(message, status_code=500)


class SearchError(DocIntelliError):
    def __init__(self, message: str = "Search operation failed"):
        super().__init__(message, status_code=500)
