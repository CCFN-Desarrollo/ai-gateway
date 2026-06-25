from app.core.config import settings
from app.services.provider_factory import build_ocr_service

identity_ocr_service = build_ocr_service(settings, pipeline_name="identity")
receipt_ocr_service = build_ocr_service(settings, pipeline_name="receipt")
csf_ocr_service = build_ocr_service(settings, pipeline_name="csf")
