import logging
from datetime import datetime, timedelta

from app.models.responses import OCRResult, RulesResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Field key aliases (English and Spanish) per field type
# ---------------------------------------------------------------------------
_DATE_KEYS = {"date", "fecha", "fecha_emision", "issue_date", "document_date", "fecha_documento"}
_TOTAL_KEYS = {
    "total",
    "amount",
    "importe",
    "monto",
    "total_amount",
    "precio_total",
    "grand_total",
    "subtotal",
}
_ISSUER_KEYS = {
    "issuer",
    "emisor",
    "company",
    "empresa",
    "nombre_empresa",
    "vendor",
    "merchant",
    "razon_social",
}
_RECEIPT_NUMBER_KEYS = {
    "receipt_number",
    "folio",
    "numero",
    "ticket",
    "invoice_number",
    "numero_factura",
    "numero_ticket",
}
_STREET_KEYS = {"street", "calle", "domicilio", "address", "direccion"}
_COLONY_KEYS = {"colony", "colonia", "neighborhood"}
_ZIP_CODE_KEYS = {"zip_code", "codigo_postal", "cp", "postal_code"}
_CITY_KEYS = {"city", "ciudad", "municipio", "town"}
_STATE_KEYS = {"state", "estado", "province"}

_NAME_KEYS = {"full_name", "nombre", "nombre_completo", "name", "apellidos", "nombre_apellidos"}
_ID_KEYS = {
    "curp",
    "id_number",
    "clave",
    "numero_identificacion",
    "folio",
    "numero",
    "id",
    "clave_elector",
}
_EXPIRY_KEYS = {
    "expiry_date",
    "vigencia",
    "fecha_vencimiento",
    "valid_until",
    "expiration",
    "fecha_expiracion",
}
_PHOTO_KEYS = {"photo", "foto", "photograph", "has_photo", "photo_present"}

_DATE_FORMATS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%Y/%m/%d",
    "%d %b %Y",
    "%B %d, %Y",
]


def _has_field(fields: dict, keys: set[str]) -> bool:
    """Return True if any of the given keys exists and has a non-empty value."""
    return any(k in fields and fields[k] for k in keys)


def _parse_date(value: str) -> datetime | None:
    """Attempt to parse a date string using multiple formats."""
    value = str(value).strip()

    # Normalize common month/year expiry formats to the last day of that month.
    if len(value) == 7 and value[4] in {"-", "/"}:
        year_str, month_str = value.split(value[4], maxsplit=1)
        if year_str.isdigit() and month_str.isdigit():
            year = int(year_str)
            month = int(month_str)
            if 1 <= month <= 12:
                if month == 12:
                    next_month = datetime(year + 1, 1, 1)
                else:
                    next_month = datetime(year, month + 1, 1)
                return next_month.replace(day=1) - timedelta(days=1)

    if len(value) == 7 and value[2] in {"-", "/"}:
        month_str, year_str = value.split(value[2], maxsplit=1)
        if year_str.isdigit() and month_str.isdigit():
            year = int(year_str)
            month = int(month_str)
            if 1 <= month <= 12:
                if month == 12:
                    next_month = datetime(year + 1, 1, 1)
                else:
                    next_month = datetime(year, month + 1, 1)
                return next_month.replace(day=1) - timedelta(days=1)

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


class RulesEngine:
    """Applies business rules to OCR-extracted data for document validation."""

    def validate_receipt(
        self, ocr_result: OCRResult, document_type: str = "RECEIPT"
    ) -> RulesResult:
        """
        Validate a receipt document against required field rules.

        Rules checked:
        - has_date: document must contain a date field
        - has_total: document must contain a total/amount field
        - has_issuer: document must contain an issuer/company name
        - has_receipt_number: document must contain a folio/ticket number

        Args:
            ocr_result: Extracted OCR data from the receipt.

        Returns:
            RulesResult with passed/failed rules and aggregate score.
        """
        fields = ocr_result.structured_fields
        passed: list[str] = []
        failed: list[str] = []

        flags: list[str] = []
        if document_type == "ADDRESS_PROOF":
            rules = [
                ("has_issue_date", _has_field(fields, _DATE_KEYS)),
                ("has_issuer", _has_field(fields, _ISSUER_KEYS)),
                ("has_street", _has_field(fields, _STREET_KEYS)),
                ("has_colony", _has_field(fields, _COLONY_KEYS)),
                ("has_zip_code", _has_field(fields, _ZIP_CODE_KEYS)),
                ("has_city", _has_field(fields, _CITY_KEYS)),
                ("has_state", _has_field(fields, _STATE_KEYS)),
            ]
        else:
            rules = [
                ("has_date", _has_field(fields, _DATE_KEYS)),
                ("has_total", _has_field(fields, _TOTAL_KEYS)),
                ("has_issuer", _has_field(fields, _ISSUER_KEYS)),
                ("has_receipt_number", _has_field(fields, _RECEIPT_NUMBER_KEYS)),
            ]

        for rule_name, passed_check in rules:
            if passed_check:
                passed.append(rule_name)
            else:
                failed.append(rule_name)

        if document_type == "ADDRESS_PROOF":
            issue_status = self.get_issue_date_freshness_status(fields)
            if issue_status == "valid":
                passed.append("issue_date_within_3_months")
            elif issue_status == "expired":
                failed.append("issue_date_within_3_months")
                flags.append("expired_document")
            else:
                flags.append("unknown_issue_date")

        rules_score = len(passed) / len(rules) if rules else 0.0
        if document_type == "ADDRESS_PROOF":
            denominator = len(passed) + len(failed)
            rules_score = len(passed) / denominator if denominator else 0.0
        logger.debug(
            "Receipt rules (%s): passed=%s failed=%s flags=%s score=%.2f",
            document_type,
            passed,
            failed,
            flags,
            rules_score,
        )
        return RulesResult(
            passed_rules=passed,
            failed_rules=failed,
            flags=flags,
            rules_score=rules_score,
        )

    def validate_identity(self, ocr_result: OCRResult, document_type: str) -> RulesResult:
        """
        Validate an identity document against required field rules.

        Rules checked:
        - has_full_name: document must contain the holder's name
        - has_id_number: document must contain CURP or ID number
        - expiry_not_past: expiry date must exist and not be in the past
        - has_photo: document must indicate presence of a photo (OCR-based hint)

        Args:
            ocr_result: Extracted OCR data from the identity document.
            document_type: Type of identity document (INE, PASAPORTE, LICENCIA).

        Returns:
            RulesResult with passed/failed rules and aggregate score.
        """
        fields = ocr_result.structured_fields
        raw_text_lower = ocr_result.raw_text.lower()
        passed: list[str] = []
        failed: list[str] = []
        flags: list[str] = []

        if document_type == "INE_REVERSO":
            if _has_field(fields, _ID_KEYS):
                passed.append("has_id_number")
            else:
                failed.append("has_id_number")

            rules_score = len(passed) / (len(passed) + len(failed)) if (passed or failed) else 0.0
            logger.debug(
                "Identity rules (%s): passed=%s failed=%s flags=%s score=%.2f",
                document_type,
                passed,
                failed,
                flags,
                rules_score,
            )
            return RulesResult(
                passed_rules=passed,
                failed_rules=failed,
                flags=flags,
                rules_score=rules_score,
            )

        # has_full_name
        if _has_field(fields, _NAME_KEYS):
            passed.append("has_full_name")
        else:
            failed.append("has_full_name")

        # has_id_number (CURP or other ID)
        if _has_field(fields, _ID_KEYS):
            passed.append("has_id_number")
        else:
            failed.append("has_id_number")

        # expiry_not_past
        expiry_status = self.get_expiry_status(fields)
        if expiry_status == "valid":
            passed.append("expiry_not_past")
        elif expiry_status == "expired":
            failed.append("expiry_not_past")
            flags.append("expired_document")
        else:
            flags.append("unknown_expiry")

        # has_photo is informational only; OCR text is not a reliable proxy for visual presence.
        photo_text_hints = {"foto", "photograph", "photo", "imagen"}
        photo_hint_in_text = any(hint in raw_text_lower for hint in photo_text_hints)
        if not (_has_field(fields, _PHOTO_KEYS) or photo_hint_in_text):
            flags.append("photo_not_detected")

        rules_score = len(passed) / (len(passed) + len(failed)) if (passed or failed) else 0.0
        logger.debug(
            "Identity rules (%s): passed=%s failed=%s flags=%s score=%.2f",
            document_type,
            passed,
            failed,
            flags,
            rules_score,
        )
        return RulesResult(
            passed_rules=passed,
            failed_rules=failed,
            flags=flags,
            rules_score=rules_score,
        )

    @staticmethod
    def get_expiry_status(fields: dict) -> str:
        """Return valid, expired, or unknown based on the OCR fields."""
        for key in _EXPIRY_KEYS:
            if key in fields and fields[key]:
                parsed = _parse_date(str(fields[key]))
                if parsed is not None:
                    return "valid" if parsed > datetime.now() else "expired"
                return "unknown"
        return "unknown"

    def get_expiry_date_str(self, fields: dict) -> str | None:
        """Return the raw expiry date string if found, else None."""
        for key in _EXPIRY_KEYS:
            if key in fields and fields[key]:
                return str(fields[key])
        return None

    @staticmethod
    def get_issue_date_freshness_status(fields: dict) -> str:
        """Return valid, expired, or unknown for issue dates older than 3 months."""
        for key in _DATE_KEYS:
            if key in fields and fields[key]:
                parsed = _parse_date(str(fields[key]))
                if parsed is not None:
                    max_age = datetime.now() - timedelta(days=92)
                    return "valid" if parsed >= max_age else "expired"
                return "unknown"
        return "unknown"

    def get_issue_date_str(self, fields: dict) -> str | None:
        """Return the raw issue date string if found, else None."""
        for key in _DATE_KEYS:
            if key in fields and fields[key]:
                return str(fields[key])
        return None


rules_engine = RulesEngine()
