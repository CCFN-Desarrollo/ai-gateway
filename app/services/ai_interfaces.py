from typing import Protocol

from app.models.responses import OCRResult, VisionResult


class OCRProvider(Protocol):
    async def extract_text(
        self,
        image_bytes: bytes,
        media_type: str = "image/jpeg",
        document_type: str | None = None,
    ) -> OCRResult: ...


class VisionProvider(Protocol):
    async def analyze_document(
        self,
        image_bytes: bytes,
        document_type: str,
        media_type: str = "image/jpeg",
    ) -> VisionResult: ...
