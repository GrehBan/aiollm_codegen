"""Schema normalization preserves values and rejects ambiguous composition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from aiollm_codegen import generate_models
from aiollm_codegen.normalization import (
    json_schema_document,
    prepare_document,
)
from pydantic import JsonValue


def test_root_repeating_its_definition_is_not_a_conflict() -> None:
    thing: dict[str, JsonValue] = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
    }
    document = json_schema_document(
        {
            "title": "Thing",
            **thing,
            "$defs": {"Thing": {"title": "A thing", **thing}},
        },
        Path("thing.schema.json"),
    )
    schemas = document["components"]
    assert isinstance(schemas, dict)
    assert schemas["schemas"] == {"Thing": {"title": "Thing", **thing}}


@pytest.mark.parametrize(
    "schema, message",
    [
        ({"$defs": []}, "definitions must be an object"),
        (
            {"title": "X", "type": "object", "$defs": {"X": {}}},
            "root conflicts",
        ),
        ({"$defs": {"X": True}}, "must be object schemas"),
    ],
)
def test_invalid_json_schema_definitions(schema: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        json_schema_document(schema, Path("root.json"))


def test_json_schema_pointer_conversion_preserves_data() -> None:
    document = json_schema_document(
        {
            "definitions": {"A": {"type": "object"}},
            "properties": {
                "self": {"$ref": "#"},
                "a": {"$ref": "#/definitions/A"},
            },
            "discriminator": {"mapping": {"a": "#/$defs/A", "b": "A", "c": 1}},
            "example": {"$ref": "#/$defs/unchanged"},
        },
        Path("root.json"),
    )
    schemas: Any = document["components"]
    root = schemas["schemas"]["Root"]
    assert root["properties"]["self"]["$ref"] == "#/components/schemas/Root"
    assert root["properties"]["a"]["$ref"] == "#/components/schemas/A"
    assert root["discriminator"]["mapping"] == {
        "a": "#/components/schemas/A",
        "b": "A",
        "c": 1,
    }
    assert root["example"]["$ref"] == "#/$defs/unchanged"


def _prepare(schemas: Any, *, request: bool = True) -> Any:
    return prepare_document(
        {"components": {"schemas": schemas}}, request=request
    )["components"]


@pytest.mark.parametrize(
    "schemas, message",
    [
        (
            {"A": {"allOf": [{"$ref": "#/components/schemas/A"}]}},
            "Cyclic allOf",
        ),
        (
            {"A": {"allOf": [{"$ref": "#/components/schemas/Missing"}]}},
            "Invalid schema reference",
        ),
        (
            {
                "A": {
                    "allOf": [{"$ref": "#/components/schemas/B/type/child"}]
                },
                "B": {"type": "string"},
            },
            "Invalid schema reference",
        ),
    ],
)
def test_invalid_composition_fails(schemas: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _prepare(schemas)


def test_object_intersections_retain_constraints_and_reference_siblings() -> (
    None
):
    result = _prepare(
        {
            "Parent": {
                "type": "object",
                "properties": {"count": {"minimum": 1}, "fixed": {"const": 2}},
            },
            "Child": {
                "allOf": [
                    {
                        "$ref": "#/components/schemas/Parent",
                        "description": "Parent fields.",
                    }
                ],
                "properties": {
                    "count": {"maximum": 3},
                    "fixed": {"type": "integer"},
                },
            },
            "Scalar": {"allOf": [True, {"type": "string"}]},
            "Object": {
                "allOf": [{"type": "object"}, {"type": "object"}],
                "properties": {"value": True},
            },
            "Overlap": {
                "allOf": [
                    {"type": "object", "properties": {"value": True}},
                    {"type": "object", "properties": {"value": False}},
                ]
            },
        }
    )["schemas"]
    # A reference sibling which is not an object retains the original allOf.
    assert result["Child"]["allOf"][1] == {"description": "Parent fields."}
    assert result["Scalar"]["allOf"][0] is True
    assert result["Overlap"]["properties"]["value"] == {"allOf": [True, False]}
    merged = _prepare(
        {
            "A": {
                "allOf": [
                    {
                        "type": "object",
                        "properties": {"n": {"const": 2}, "x": {"minimum": 1}},
                    },
                    {
                        "type": "object",
                        "properties": {
                            "n": {"type": "integer"},
                            "x": {"maximum": 3},
                        },
                    },
                ]
            }
        }
    )["schemas"]["A"]["properties"]
    assert merged["n"] == {"type": "integer", "const": 2}
    assert merged["x"] == {"allOf": [{"minimum": 1}, {"maximum": 3}]}


@pytest.mark.parametrize(
    "variants, discriminator, retained",
    [
        ([{}], {"propertyName": "type"}, False),
        ([True], {"propertyName": "type"}, False),
        (
            [{"$ref": "#/components/schemas/A"}] * 2,
            {"propertyName": "type"},
            False,
        ),
        (
            [{"$ref": "#/components/schemas/NoTag"}],
            {"propertyName": "type"},
            False,
        ),
        ([{"$ref": "#/components/schemas/A"}], {"propertyName": 1}, True),
    ],
)
def test_ambiguous_discriminators_do_not_invent_tags(
    variants: Any,
    discriminator: Any,
    retained: bool,
) -> None:
    result = _prepare(
        {
            "A": {"properties": {"type": {"enum": ["a"]}}},
            "NoTag": {"properties": {"type": {"enum": ["a", "b"]}}},
            "Union": {"oneOf": variants, "discriminator": discriminator},
        }
    )["schemas"]["Union"]
    assert ("discriminator" in result) is retained


def test_inherited_tag_lookup_and_reference_cycles() -> None:
    result = _prepare(
        {
            "Cycle": {"$ref": "#/components/schemas/Cycle"},
            "Inherited": {
                "allOf": [True, {}, {"properties": {"type": {"enum": ["a"]}}}]
            },
            "Empty": {"allOf": [True, {}]},
            "Numeric": {"properties": {"type": {"enum": [1]}}},
            **{
                name + "Union": {
                    "oneOf": [{"$ref": "#/components/schemas/" + name}],
                    "discriminator": {"propertyName": "type"},
                }
                for name in ("Cycle", "Inherited", "Empty", "Numeric")
            },
        }
    )["schemas"]
    assert result["InheritedUnion"]["discriminator"]["mapping"] == {
        "a": "#/components/schemas/Inherited"
    }
    for name in ("Cycle", "Empty", "Numeric"):
        assert "discriminator" not in result[name + "Union"]


def test_nullable_response_vocabulary_and_shared_enum() -> None:
    result = _prepare(
        {
            "Status": {"enum": ["ready"]},
            "Result": {
                "properties": {
                    "status": {"$ref": "#/components/schemas/Status"},
                    "stop_reason": {"enum": [None, "stop"]},
                    "code": {"const": "busy"},
                }
            },
        },
        request=False,
    )["schemas"]
    assert result["Status"] == {"type": "string"}
    assert result["Result"]["properties"]["stop_reason"]["type"] == [
        "string",
        "null",
    ]
    assert result["Result"]["properties"]["code"] == {"type": "string"}


def _generate_literal(
    source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> str:
    path = tmp_path / "schema.json"
    path.write_text('{"type": "string"}', encoding="utf-8")

    def backend(*args: Any, **kwargs: Any) -> str:
        kwargs["emit_model_metadata"].write_text(
            '{"models": []}', encoding="utf-8"
        )
        return source

    monkeypatch.setattr("aiollm_codegen.generator.generate", backend)
    return generate_models(path, kind="request")


@pytest.mark.parametrize(
    "expression",
    ["Literal[str]", "Literal[list[str]]", "Literal[1.2]", "Literal[{}]"],
)
def test_literal_annotations_reject_types_and_nonliteral_values(
    expression: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="Literal arguments must be real"):
        _generate_literal(
            "from typing import Literal\ntype Value = " + expression,
            tmp_path,
            monkeypatch,
        )


def test_literal_enum_members_remain_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = _generate_literal(
        "from enum import StrEnum\nfrom typing import Literal\n"
        "class Mode(StrEnum):\n    auto = 'auto'\n"
        "type Value = Literal[Mode.auto]\n",
        tmp_path,
        monkeypatch,
    )
    assert "Literal[Mode.auto]" in code


def test_length_limited_urls_keep_their_limit(spec: Any, models: Any) -> None:
    source = spec(
        {
            "Image": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "format": "uri", "maxLength": 30}
                },
            }
        }
    )
    module = models(source, kind="request")
    assert module.Image(url="data:image/png;base64,AAAA").url.startswith(
        "data:"
    )
    with pytest.raises(ValueError, match="at most 30"):
        module.Image(url="https://example.com/" + "x" * 30)


def test_responses_accept_absent_nullable_fields(
    spec: Any, models: Any
) -> None:
    source = spec(
        {
            "Details": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "Reply": {
                "type": "object",
                "required": ["id", "note", "details", "count"],
                "properties": {
                    "id": {"type": "string"},
                    "note": {"type": ["string", "null"]},
                    "details": {"$ref": "#/components/schemas/Details"},
                    "count": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}]
                    },
                },
            },
        }
    )
    reply = models(source, kind="response").Reply.model_validate({"id": "r"})
    assert (reply.note, reply.details, reply.count) == (None, None, None)
    with pytest.raises(ValueError, match="id"):
        models(source, kind="response").Reply.model_validate({})
    request = models(source, kind="request").Reply
    with pytest.raises(ValueError, match="note"):
        request.model_validate({"id": "r", "details": None, "count": None})
