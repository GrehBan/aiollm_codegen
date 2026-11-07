"""Generated annotations, metadata and uploads follow aiollm contracts."""

from __future__ import annotations

import ast
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from aiollm_codegen import generate_models
from pydantic import TypeAdapter, ValidationError

from aiollm.types.files import BytesFile, InputFile


@pytest.mark.parametrize("extension", ["json", "yaml"])
def test_json_schema_contract_input_and_field_metadata(
    tmp_path: Path, models: Callable[..., Any], extension: str
) -> None:
    schema = {
        "title": "Request",
        "type": "object",
        "required": ["kind", "enabled", "count"],
        "$defs": {
            "Nested": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
            }
        },
        "properties": {
            "kind": {
                "const": "str",
                "description": "\n Kind.\\n\\n Details.\n",
            },
            "enabled": {"const": True},
            "count": {"const": -2},
            "nested": {"$ref": "#/$defs/Nested"},
        },
    }
    path = tmp_path / f"request.schema.{extension}"
    path.write_text(
        json.dumps(schema) if extension == "json" else yaml.safe_dump(schema),
        encoding="utf-8",
    )
    module = models(path, kind="request")
    request = module.Request(
        kind="str", enabled=True, count=-2, nested={"text": "Hi"}
    )
    assert request.kind == "str"
    assert request.nested.text == "Hi"
    assert (
        module.Request.model_fields["kind"].description == "Kind.\n\nDetails."
    )
    assert (
        module.Request.model_json_schema()["properties"]["kind"]["description"]
        == "Kind.\n\nDetails."
    )
    with pytest.raises(ValidationError):
        module.Request(kind="int", enabled=True, count=-2)
    tree = ast.parse(generate_models(path, kind="request"))
    model = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Request"
    )
    assert not any(
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and "Kind." in str(node.value.value)
        for node in model.body[1:]
    )


def test_binary_request_fields_use_aiollm_input_file(
    spec: Callable[..., Path], models: Callable[..., Any]
) -> None:
    source = spec(
        {
            "Upload": {
                "type": "object",
                "required": ["file"],
                "properties": {
                    "file": {
                        "type": "string",
                        "format": "binary",
                        "description": "The uploaded file.",
                    }
                },
            }
        }
    )
    module = models(source, kind="request")
    upload = module.Upload(file=b"data")
    assert module.Upload.model_fields["file"].annotation is InputFile
    assert isinstance(upload.file, BytesFile)
    assert (
        module.Upload.model_fields["file"].description == "The uploaded file."
    )


def test_inherited_discriminator_values_are_wire_values(
    spec: Callable[..., Path], models: Callable[..., Any]
) -> None:
    source = spec(
        {
            "Base": {
                "type": "object",
                "required": ["type"],
                "properties": {
                    "type": {"const": "wire-tag"},
                    "status": {"type": "string"},
                },
            },
            "Child": {
                "allOf": [{"$ref": "#/components/schemas/Base"}],
                "required": ["status"],
                "properties": {
                    "status": {"type": "string", "enum": ["ready", "done"]}
                },
            },
            "Other": {
                "type": "object",
                "required": ["type"],
                "properties": {"type": {"const": "other"}},
            },
            "Result": {
                "oneOf": [
                    {"$ref": "#/components/schemas/Child"},
                    {"$ref": "#/components/schemas/Other"},
                ],
                "discriminator": {"propertyName": "type"},
            },
        }
    )
    module = models(source, kind="response")
    child = module.Child(type="wire-tag", status="ready")
    assert child.type == "wire-tag"
    assert module.Child.__bases__ == (module.Other.__bases__[0],)


def test_responses_accept_new_values_but_requests_keep_enums(
    spec: Callable[..., Path], models: Callable[..., Any]
) -> None:
    source = spec(
        {
            "Result": {
                "type": "object",
                "required": ["status", "stop_reason", "error_code", "type"],
                "properties": {
                    "status": {"type": "string", "enum": ["ready", "done"]},
                    "stop_reason": {"enum": ["stop"]},
                    "error_code": {"type": "string", "enum": ["bad", "busy"]},
                    "type": {"const": "result"},
                },
            }
        }
    )
    response = models(source, kind="response")
    value = response.Result(
        status="future",
        stop_reason="future",
        error_code="future",
        type="result",
    )
    assert value.status == value.stop_reason == value.error_code == "future"
    for field in ("status", "stop_reason", "error_code"):
        assert response.Result.model_fields[field].annotation is str
    with pytest.raises(ValidationError):
        response.Result(
            status="ready", stop_reason="stop", error_code="bad", type="wrong"
        )
    request = models(source, kind="request")
    with pytest.raises(ValidationError):
        request.Result(
            status="future",
            stop_reason="stop",
            error_code="bad",
            type="result",
        )


@pytest.mark.parametrize("discriminated", [True, False])
def test_unknown_response_variants_keep_data_and_known_variants_stay_strict(
    spec: Callable[..., Path], models: Callable[..., Any], discriminated: bool
) -> None:
    union: dict[str, Any] = {
        "oneOf": [
            {"$ref": "#/components/schemas/TextEvent"},
            {"$ref": "#/components/schemas/DoneEvent"},
        ]
    }
    if discriminated:
        union["discriminator"] = {"propertyName": "type"}
    source = spec(
        {
            "TextEvent": {
                "type": "object",
                "required": ["type", "text"],
                "properties": {
                    "type": {"const": "text"},
                    "text": {"type": "string"},
                },
            },
            "DoneEvent": {
                "type": "object",
                "required": ["type"],
                "properties": {"type": {"const": "done"}},
            },
            "Event": union,
        }
    )
    module = models(source, kind="response")
    adapter = TypeAdapter(module.Event)
    unknown = adapter.validate_python({"type": "future", "payload": {"x": 1}})
    assert unknown.type == "future"
    assert unknown.model_extra == {"payload": {"x": 1}}
    assert unknown.model_dump() == {"type": "future", "payload": {"x": 1}}
    assert isinstance(
        adapter.validate_python({"type": "text", "text": "hi"}),
        module.TextEvent,
    )
    for invalid in ({"type": "text"}, {"type": 12}, {"payload": 1}):
        with pytest.raises(ValidationError):
            adapter.validate_python(invalid)
    assert "Generated provider schema type." not in module.TextEvent.__doc__
    # No field is described upstream, so the class lists none.
    assert module.TextEvent.__doc__.startswith("`TextEvent` from")
    assert "Attributes:" not in module.TextEvent.__doc__
    assert "text:" not in module.TextEvent.__doc__


def test_shared_known_tags_and_nullable_union_keep_their_contracts(
    spec: Callable[..., Path],
    models: Callable[..., Any],
) -> None:
    source = spec(
        {
            "UnknownVariant": {"type": "object"},
            "A": {
                "type": "object",
                "required": ["type", "text"],
                "properties": {
                    "type": {"const": "same"},
                    "text": {"type": "string"},
                },
            },
            "B": {
                "type": "object",
                "required": ["type", "image"],
                "properties": {
                    "type": {"const": "same"},
                    "image": {"type": "string"},
                },
            },
            "Union": {
                "anyOf": [
                    {"$ref": "#/components/schemas/A"},
                    {"$ref": "#/components/schemas/B"},
                    {"type": "null"},
                ]
            },
        }
    )
    module = models(source, kind="response")
    adapter = TypeAdapter(module.Union)
    assert adapter.validate_python(None) is None
    assert adapter.validate_python({"type": "same", "text": "hi"}).text == "hi"
    assert (
        adapter.validate_python({"type": "same", "image": "url"}).image
        == "url"
    )
    assert adapter.validate_python(
        {"type": "unknown", "x": 1}
    ).model_extra == {"x": 1}
    with pytest.raises(ValidationError):
        adapter.validate_python({"type": "same"})
