import pytest

from app.core.errors import ProviderResponseError
from app.services.provider_common import parse_json_response


def test_parse_plain_json_object() -> None:
    data = parse_json_response(
        '{"raw_text":"hello","structured_fields":{},"confidence":0.9}',
        "bad json",
    )
    assert data["raw_text"] == "hello"
    assert data["confidence"] == 0.9


def test_parse_markdown_fenced_json() -> None:
    payload = """```json
{"raw_text": "from fence", "structured_fields": {"full_name": "Ada"}, "confidence": 0.8}
```"""
    data = parse_json_response(payload, "bad json")
    assert data["raw_text"] == "from fence"
    assert data["structured_fields"]["full_name"] == "Ada"


def test_parse_json_with_surrounding_prose() -> None:
    payload = (
        "Here is the extraction result:\n"
        '{"raw_text":"partial","structured_fields":{"id_number":"X1"},"confidence":0.7}\n'
        "Hope that helps!"
    )
    data = parse_json_response(payload, "bad json")
    assert data["structured_fields"]["id_number"] == "X1"


def test_parse_rejects_non_object_json() -> None:
    with pytest.raises(ProviderResponseError, match="bad json"):
        parse_json_response('["not", "an", "object"]', "bad json")


def test_parse_rejects_empty_or_garbage() -> None:
    with pytest.raises(ProviderResponseError, match="bad json"):
        parse_json_response("", "bad json")
    with pytest.raises(ProviderResponseError, match="bad json"):
        parse_json_response("definitely not json {", "bad json")


def test_parse_detects_provider_refusal() -> None:
    refusal = (
        "I'm not able to extract or analyze personal identification information "
        "from identity documents. This appears to be a US border crossing card."
    )
    with pytest.raises(ProviderResponseError, match="refused"):
        parse_json_response(refusal, "bad json")
