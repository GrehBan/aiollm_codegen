"""Input validation, reference closure, and bounded schema traversal."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import pytest
import yaml
from aiollm_codegen import generate_models
from aiollm_codegen.schema import load_schemas, schema_nodes
from pydantic import JsonValue


@pytest.mark.parametrize("kind", ["request", "response"])
def test_yaml_dates_stay_strings_in_examples_enums_and_defaults(
    tmp_path: Path,
    models: Callable[..., Any],
    kind: Literal["request", "response"],
) -> None:
    path = tmp_path / "dates.openapi.yaml"
    path.write_text(
        """\
openapi: 3.1.0
info:
  title: Timestamp regression
  version: 1.0.0
paths:
  /example:
    get:
      responses:
        '200':
          description: OK
          content:
            application/json:
              example:
                version: 2024-10-01
components:
  schemas:
    DatedRequest:
      type: object
      required: [model]
      properties:
        model:
          type: string
          enum: [2026-10-08, latest]
        version:
          type: string
          default: 2024-10-01
        timestamp:
          type: string
          default: 2024-10-01T12:34:56Z
        count:
          type: integer
          default: 3
        enabled:
          type: boolean
          default: true
        nullable:
          type: [string, 'null']
          default: null
      example:
        version: 2024-10-01
        timestamp: 2024-10-01T12:34:56Z
""",
        encoding="utf-8",
    )
    result: Any = load_schemas(path, request=False)
    schema = result["components"]["schemas"]["DatedRequest"]
    assert schema["properties"]["model"]["enum"] == ["2026-10-08", "latest"]
    assert schema["example"] == {
        "version": "2024-10-01",
        "timestamp": "2024-10-01T12:34:56Z",
    }
    assert schema["properties"]["version"]["default"] == "2024-10-01"
    assert schema["properties"]["count"]["default"] == 3
    assert schema["properties"]["enabled"]["default"] is True
    assert schema["properties"]["nullable"]["default"] is None
    module = models(path, kind=kind)
    request = module.DatedRequest(model="2026-10-08")
    assert request.model_dump(mode="json")["model"] == "2026-10-08"
    if kind == "response":
        assert request.version == "2024-10-01"
        assert request.timestamp == "2024-10-01T12:34:56Z"
    # Loading OpenAPI must not change PyYAML's behavior for other callers.
    assert yaml.safe_load("version: 2024-10-01") == {
        "version": date(2024, 10, 1)
    }


@pytest.mark.parametrize("content", ["[]", "openapi: 2.0", "openapi: 3.1.0"])
def test_invalid_document(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="Expected"):
        load_schemas(path, request=True)


def test_unknown_and_non_object_components(spec: Callable[..., Path]) -> None:
    path = spec({"Valid": {"type": "object"}, "Invalid": True})
    with pytest.raises(ValueError, match="Unknown component schema"):
        generate_models(path, kind="request", schemas=["Missing"])
    with pytest.raises(ValueError, match="must be an object"):
        generate_models(path, kind="request", schemas=["Invalid"])


@pytest.mark.parametrize(
    "reference", ["other.yaml#/A", "https://example.test/A", "#/other", 2]
)
def test_external_and_non_component_refs_rejected(
    spec: Callable[..., Path], reference: Any
) -> None:
    path = spec({"A": {"$ref": reference}})
    with pytest.raises(ValueError, match="Bundle references"):
        generate_models(path, kind="request")


@pytest.mark.parametrize("keyword", ["$id", "$dynamicRef", "$recursiveRef"])
def test_unsupported_reference_scopes_rejected(
    spec: Callable[..., Path], keyword: str
) -> None:
    path = spec({"A": {"type": "object", keyword: "elsewhere"}})
    with pytest.raises(ValueError, match="Dynamic references"):
        generate_models(path, kind="request")


def test_schema_keywords_are_walked_but_payloads_are_preserved(
    spec: Callable[..., Path],
) -> None:
    path = spec(
        {
            "A": {
                "type": "object",
                "properties": {"default": {"type": "string", "default": "x"}},
                "patternProperties": {
                    ".*": {"type": "string", "default": "y"}
                },
                "additionalProperties": {"$ref": "#/components/schemas/B"},
                "allOf": [{"type": "object", "default": {"data": 2}}],
                "examples": [{"$ref": "literal value", "default": "keep"}],
            },
            "B": {"type": "string"},
            "Unused": {"type": "string"},
        }
    )
    result: Any = load_schemas(path, request=True, schemas=["A"])
    schemas = result["components"]["schemas"]
    assert set(schemas) == {"A", "B"}
    assert schemas["A"]["properties"]["default"] == {"type": "string"}
    assert schemas["A"]["examples"] == [
        {"$ref": "literal value", "default": "keep"}
    ]
    assert "default" not in schemas["A"]["patternProperties"][".*"]
    assert "default" not in schemas["A"]["allOf"][0]


def test_reference_pointer_escaping(
    spec: Callable[..., Path], models: Callable[..., Any]
) -> None:
    path = spec(
        {
            "Container": {
                "type": "object",
                "properties": {"value": {"$ref": "#/components/schemas/A~1B"}},
            },
            "A/B": {
                "type": "object",
                "properties": {"n": {"type": "integer"}},
            },
        }
    )
    module = models(path, kind="request", schemas=["Container"])
    instance = module.Container(value={"n": 1})
    assert instance.value.n == 1
    assert instance.value.__upstream_schema__ == "A/B"


def test_oversized_input_rejected(tmp_path: Path) -> None:
    path = tmp_path / "large.yaml"
    with path.open("wb") as stream:
        stream.truncate(32 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="32 MiB"):
        load_schemas(path, request=True)


def test_schema_traversal_limit() -> None:
    children: list[JsonValue] = [{"type": "string"}]
    schema: dict[str, JsonValue] = {"allOf": children * 100_000}
    with pytest.raises(ValueError, match="100,000-node"):
        list(schema_nodes(schema))


def test_bad_kind_is_rejected(spec: Callable[..., Path]) -> None:
    path = spec({"A": {"type": "object"}})
    invalid: Any = "invalid"
    with pytest.raises(ValueError, match="kind must be"):
        generate_models(path, kind=invalid)


def test_multiple_module_backend_result_rejected(
    spec: Callable[..., Path],
) -> None:
    path = spec({"A": {"type": "object"}})

    def generate(*args: Any, **kwargs: Any) -> Any:
        kwargs["emit_model_metadata"].write_text(
            '{"models": []}', encoding="utf-8"
        )
        return {}

    with (
        patch("aiollm_codegen.generator.generate", side_effect=generate),
        pytest.raises(ValueError, match="single generated module"),
    ):
        generate_models(path, kind="request")
