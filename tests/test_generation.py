"""Observable generated-model contracts, including actual aiollm methods."""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

import pytest
from aiollm_codegen import generate_models
from conftest import EXAMPLE
from pydantic import MISSING, TypeAdapter, ValidationError

from aiollm import BaseAPIMethod, BaseAPIType, BaseAPITypeExtra, BodyFormat


def _is_subclass(value: Any, base: type[Any]) -> bool:
    return issubclass(value, base)


def test_backend_forward_aliases_are_completed_before_subclassing(
    models: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def backend(*args: Any, **kwargs: Any) -> str:
        kwargs["emit_model_metadata"].write_text(
            '{"models": []}', encoding="utf-8"
        )
        return (
            "from __future__ import annotations\n"
            "from pydantic import MISSING\n"
            "from aiollm.types.base import BaseAPIType\n"
            "class Params(BaseAPIType):\n"
            "    metadata: Metadata | MISSING = MISSING\n"
            "type Metadata = dict[str, str] | None\n"
        )

    monkeypatch.setattr("aiollm_codegen.generator.generate", backend)
    module = models(kind="request")
    assert module.Params.__pydantic_complete__
    method: Any = type(
        "Create",
        (module.Params, BaseAPIMethod[str]),
        {
            "__request_path__": "/generation",
            "__request_method__": "POST",
            "__request_body_format__": BodyFormat.JSON,
        },
    )
    assert method(metadata={"key": "value"}).get_body_map() == {
        "metadata": {"key": "value"}
    }


@pytest.mark.parametrize("kind", ["request", "response"])
def test_docstrings_render_newlines_and_trim_blank_edges(
    models: Callable[..., Any],
    spec: Callable[..., Path],
    kind: Literal["request", "response"],
) -> None:
    source = spec(
        {
            "Described": {
                "type": "object",
                "description": (
                    "\n\n  Summary.\\n\\n  Details with Unicode: café.\n\n  "
                ),
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "\n  Field summary.\n\n  Field details.\n\n  "
                        ),
                        "default": r"keep\nthis\path",
                    },
                    "value": {"$ref": "#/components/schemas/Alias"},
                },
            },
            "Alias": {
                "type": "string",
                "description": "\n\n Alias summary.\\n Alias details.\n\n ",
            },
        }
    )
    code = generate_models(source, kind=kind)
    tree = ast.parse(code)
    described = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Described"
    )
    description = ast.get_docstring(described)
    assert description is not None
    assert (
        description.split("\n\nAttributes:", 1)[0]
        == "Summary.\n\nDetails with Unicode: café."
    )
    assert not any(
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and "Field summary." in str(node.value.value)
        for node in described.body[1:]
    )
    module = models(source, kind=kind)
    assert (
        TypeAdapter(module.Alias).json_schema()["description"]
        == "Alias summary.\nAlias details."
    )
    assert (
        module.Described.model_fields["text"].description
        == "Field summary.\n\nField details."
    )
    assert inspect.getdoc(module.Described) == description
    if kind == "response":
        assert module.Described().text == r"keep\nthis\path"


@pytest.mark.parametrize("kind", ["request", "response"])
def test_root_schemas_are_reusable_type_aliases(
    models: Callable[..., Any],
    spec: Callable[..., Path],
    kind: Literal["request", "response"],
) -> None:
    source = spec(
        {
            "Message": {
                "type": "object",
                "required": ["type", "text"],
                "properties": {
                    "type": {"const": "message"},
                    "text": {"type": "string"},
                },
            },
            "Tool": {
                "type": "object",
                "required": ["type", "count"],
                "properties": {
                    "type": {"const": "tool"},
                    "count": {"type": "integer"},
                },
            },
            "InputItem": {
                "oneOf": [
                    {"$ref": "#/components/schemas/Message"},
                    {"$ref": "#/components/schemas/Tool"},
                ],
                "discriminator": {"propertyName": "type"},
            },
            "InputItems": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/InputItem"},
            },
            "LimitedText": {"type": "string", "minLength": 2, "maxLength": 5},
            "Recursive": {
                "anyOf": [
                    {"type": "string"},
                    {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/Recursive"},
                    },
                ]
            },
            "Holder": {
                "type": "object",
                "required": ["items", "prompt", "tree"],
                "properties": {
                    "items": {"$ref": "#/components/schemas/InputItems"},
                    "prompt": {"$ref": "#/components/schemas/LimitedText"},
                    "tree": {"$ref": "#/components/schemas/Recursive"},
                },
            },
        }
    )
    code = generate_models(source, kind=kind)
    tree = ast.parse(code)
    aliases = {
        node.name.id for node in tree.body if isinstance(node, ast.TypeAlias)
    }
    assert {"InputItem", "InputItems", "LimitedText", "Recursive"} <= aliases
    assert "RootModel" not in code
    module = models(source, kind=kind)
    adapter = TypeAdapter(module.InputItem)
    item = adapter.validate_python({"type": "message", "text": "Hi"})
    assert isinstance(item, module.Message)
    assert item.text == "Hi"
    with pytest.raises(ValidationError):
        adapter.validate_python({"type": "tool", "text": "wrong"})
    payload = {
        "items": [{"type": "message", "text": "Hi"}],
        "prompt": "hello",
        "tree": ["a", ["b"]],
    }
    holder = module.Holder.model_validate(payload)
    assert isinstance(holder.items, list)
    assert isinstance(holder.items[0], module.Message)
    assert holder.model_dump(mode="json", by_alias=True) == payload
    for prompt in ("x", "too long"):
        with pytest.raises(ValidationError):
            module.Holder.model_validate({**payload, "prompt": prompt})
    with pytest.raises(ValidationError):
        module.Holder.model_validate({**payload, "tree": {"not": "a tree"}})


@pytest.mark.parametrize("kind", ["request", "response"])
def test_single_value_enums_are_literals(
    models: Callable[..., Any],
    spec: Callable[..., Path],
    kind: Literal["request", "response"],
) -> None:
    source = spec(
        {
            "Choice": {
                "type": "object",
                "required": ["policy", "mode"],
                "properties": {
                    "policy": {"type": "string", "enum": ["always"]},
                    "mode": {"type": "string", "enum": ["auto", "manual"]},
                    "optionalPolicy": {"type": "string", "enum": ["always"]},
                },
            },
        }
    )
    module = models(source, kind=kind)
    field = module.Choice.model_fields["policy"]
    assert get_origin(field.annotation) is Literal
    assert get_args(field.annotation) == ("always",)
    assert field.is_required() is (kind == "response")
    choice = module.Choice(policy="always", mode="auto")
    assert choice.policy == "always"
    assert choice.mode == "auto"
    if kind == "request":
        # Closed request vocabularies are Literal values, as in aiollm.
        mode = module.Choice.model_fields["mode"].annotation
        assert get_origin(mode) is Literal
        assert get_args(mode) == ("auto", "manual")
    if kind == "request":
        assert choice.optional_policy is MISSING
    assert choice.model_dump(mode="json", exclude_unset=True) == {
        "policy": "always",
        "mode": "auto",
    }
    for payload in ({"mode": "auto"}, {"policy": "never", "mode": "auto"}):
        if kind == "request" and "policy" not in payload:
            # A request fills a single-value tag and still sends it.
            filled = module.Choice.model_validate(payload)
            assert filled.model_dump()["policy"] == "always"
            continue
        with pytest.raises(ValidationError):
            module.Choice.model_validate(payload)
    assert not hasattr(module, "Policy")


def test_request_omission_null_constraints_aliases_and_method(
    models: Callable[..., Any],
) -> None:
    module = models(kind="request", schemas=["CreateRequest"])
    assert _is_subclass(module.CreateRequest, BaseAPIType)
    assert not hasattr(module, "GenerationResponse")
    request = module.CreateRequest(
        model="test",
        messages=[{"role": "user", "content": "Hi"}],
        nullable_required=None,
    )
    assert request.max_tokens is MISSING
    assert request.optional_nullable is MISSING
    assert request.model_dump(by_alias=True) == {
        "model": "test",
        "messages": [{"role": "user", "content": "Hi"}],
        "nullableRequired": None,
    }
    request = module.CreateRequest.model_validate(
        {
            "model": "test",
            "messages": [],
            "nullableRequired": None,
            "optionalNullable": None,
            "max-tokens": 3,
        }
    )
    assert request.max_tokens == 3
    assert request.model_dump(by_alias=True)["optionalNullable"] is None
    with pytest.raises(ValidationError):
        module.CreateRequest(model="test", messages=[])
    for value in (None, 0):
        with pytest.raises(ValidationError):
            module.CreateRequest(
                model="test",
                messages=[],
                nullable_required=None,
                max_tokens=value,
            )
    method: Any = type(
        "Create",
        (module.CreateRequest, BaseAPIMethod[BaseAPITypeExtra]),
        {
            "__request_path__": "/generation",
            "__request_method__": "POST",
            "__request_body_format__": BodyFormat.JSON,
        },
    )
    instance = method(model="test", messages=[], nullable_required=None)
    assert instance.get_body_map() == {
        "model": "test",
        "messages": [],
        "nullableRequired": None,
    }
    assert module.CreateRequest.__upstream_schema__ == "CreateRequest"
    assert module.Message.__upstream_schema__ == "Message"
    role = module.Message.model_fields["role"].annotation
    assert get_args(role) == ("user", "assistant")


def test_response_preserves_unknown_fields_and_wire_names(
    models: Callable[..., Any],
) -> None:
    module = models(kind="response", schemas=["GenerationResponse"])
    assert _is_subclass(module.GenerationResponse, BaseAPITypeExtra)
    response = module.GenerationResponse.model_validate(
        {
            "id": "gen-1",
            "output": "Hello",
            "new-field": {"a": 1},
            "usage": {"output-tokens": 2, "new-counter": 3},
        }
    )
    assert response.model_extra == {"new-field": {"a": 1}}
    assert response.usage.output_tokens == 2
    assert response.usage.model_extra == {"new-counter": 3}
    assert response.model_dump(by_alias=True)["usage"]["output-tokens"] == 2
    assert module.GenerationResponse(id="1", output="ok").usage is None
    with pytest.raises(ValidationError):
        module.GenerationResponse(id="1")


def test_response_models_inherit_extra_policy_from_base() -> None:
    code = generate_models(EXAMPLE, kind="response")
    assert "extra=" not in code
    assert "ConfigDict" not in code


def test_response_config_keeps_settings_other_than_extra(
    models: Callable[..., Any], spec: Callable[..., Path]
) -> None:
    source = spec(
        {
            "Thing": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "pattern": "^(?!x)"}
                },
                "additionalProperties": False,
            }
        }
    )
    code = generate_models(source, kind="response")
    assert "regex_engine" in code
    assert "extra=" not in code
    thing = models(source, kind="response").Thing.model_validate(
        {"code": "ab", "added": 1}
    )
    assert thing.model_extra == {"added": 1}


@pytest.mark.parametrize("kind", ["request", "response"])
def test_nullable_definitions_show_none_on_every_field(
    models: Callable[..., Any],
    spec: Callable[..., Path],
    kind: Literal["request", "response"],
) -> None:
    source = spec(
        {
            "Metadata": {
                "anyOf": [
                    {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    {"type": "null"},
                ]
            },
            "Thing": {
                "type": "object",
                "required": ["tags"],
                "properties": {
                    "metadata": {"$ref": "#/components/schemas/Metadata"},
                    "tags": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/Metadata"},
                    },
                },
            },
        }
    )
    module = models(source, kind=kind)
    fields = module.Thing.model_fields
    assert type(None) in get_args(fields["metadata"].annotation)
    (item,) = get_args(fields["tags"].annotation)
    assert type(None) in get_args(item)
    thing = module.Thing.model_validate({"metadata": None, "tags": [None]})
    assert thing.metadata is None and thing.tags == [None]


def test_generation_is_deterministic() -> None:
    first = generate_models(EXAMPLE, kind="request")
    assert first == generate_models(EXAMPLE, kind="request")
    assert "alias=" not in first.replace("validation_alias=", "").replace(
        "serialization_alias=", ""
    )
    ast.parse(first)


@pytest.mark.parametrize("kind", ["request", "response"])
def test_recursive_models_allof_and_discriminated_union(
    models: Callable[..., Any],
    spec: Callable[..., Path],
    kind: Literal["request", "response"],
) -> None:
    source = spec(
        {
            "Base": {
                "type": "object",
                "required": ["id"],
                "properties": {"id": {"type": "string"}},
            },
            "Tree": {
                "allOf": [
                    {"$ref": "#/components/schemas/Base"},
                    {
                        "type": "object",
                        "properties": {
                            "child": {"$ref": "#/components/schemas/Tree"}
                        },
                    },
                ]
            },
            "Text": {
                "type": "object",
                "required": ["type", "text"],
                "properties": {
                    "type": {"const": "text"},
                    "text": {"type": "string"},
                },
            },
            "Number": {
                "type": "object",
                "required": ["type", "value"],
                "properties": {
                    "type": {"const": "number"},
                    "value": {"type": "integer"},
                },
            },
            "Result": {
                "oneOf": [
                    {"$ref": "#/components/schemas/Text"},
                    {"$ref": "#/components/schemas/Number"},
                ],
                "discriminator": {"propertyName": "type"},
            },
            "Inline": {
                "type": "object",
                "properties": {
                    "payload": {
                        "type": "object",
                        "properties": {"value": {"type": "integer"}},
                    }
                },
            },
        }
    )
    module = models(source, kind=kind)
    tree = module.Tree(id="a", child={"id": "b"})
    assert tree.child.id == "b"
    adapter = TypeAdapter(module.Result)
    result = adapter.validate_python({"type": "number", "value": 2})
    assert result.value == 2
    with pytest.raises(ValidationError):
        adapter.validate_python({"type": "text", "value": 2})
    assert (
        module.Inline(payload={"value": 2}).payload.__upstream_schema__ == ()
    )


def test_openapi_30_nullable_and_schema_names(
    models: Callable[..., Any], spec: Callable[..., Path]
) -> None:
    source = spec(
        {
            "wire-name": {
                "type": "object",
                "required": ["maybe"],
                "properties": {
                    "maybe": {"type": "string", "nullable": True},
                    "default": {"type": "string"},
                },
            },
        },
        version="3.0.3",
    )
    module = models(source, kind="request")
    assert module.WireName(maybe=None).default is MISSING
    assert module.WireName.__upstream_schema__ == "wire-name"


def test_object_parent_name_ending_in_enum_keeps_child_identity(
    models: Callable[..., Any], spec: Callable[..., Path]
) -> None:
    source = spec(
        {
            "BaseEnum": {
                "type": "object",
                "properties": {"n": {"type": "integer"}},
            },
            "Child": {
                "allOf": [
                    {"$ref": "#/components/schemas/BaseEnum"},
                    {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                    },
                ]
            },
        }
    )
    module = models(source, kind="request")
    assert module.Child.__upstream_schema__ == "Child"


def test_requests_reject_unknown_keys_and_fill_single_tags(
    models: Callable[..., Any], spec: Callable[..., Path]
) -> None:
    source = spec(
        {
            "ToolChoiceNone": {
                "type": "object",
                "required": ["type"],
                "properties": {"type": {"const": "none"}},
            },
            "FunctionTool": {
                "type": "object",
                "required": ["type", "name"],
                "properties": {
                    "type": {"type": "string", "enum": ["function"]},
                    "name": {"type": "string"},
                },
            },
        }
    )
    request = models(source, kind="request")
    with pytest.raises(ValidationError, match="disable_parallel_tool_use"):
        request.ToolChoiceNone.model_validate(
            {"type": "none", "disable_parallel_tool_use": True}
        )
    with pytest.raises(ValidationError, match="bogus"):
        request.FunctionTool(name="f", bogus=1)
    tool = request.FunctionTool(name="f")
    assert tool.model_dump(by_alias=True) == {"type": "function", "name": "f"}
    response = models(source, kind="response")
    kept = response.ToolChoiceNone.model_validate({"type": "none", "x": 1})
    assert kept.model_extra == {"x": 1}


def test_request_objects_stay_open_where_the_schema_is_open(
    models: Callable[..., Any], spec: Callable[..., Path]
) -> None:
    source = spec(
        {
            "JsonSchema": {"type": "object", "additionalProperties": True},
            "Bare": {"type": "object"},
            "Open": {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "additionalProperties": True,
            },
            "Closed": {
                "type": "object",
                "properties": {"a": {"type": "string"}},
            },
            "Wrapper": {
                "anyOf": [
                    {
                        "type": "object",
                        "title": "Item",
                        "$ref": "#/components/schemas/Closed",
                    },
                    {"type": "string"},
                ]
            },
        }
    )
    module = models(source, kind="request")
    with pytest.raises(ValidationError, match="b"):
        TypeAdapter(module.Wrapper).validate_python({"a": "1", "b": 2})
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    assert TypeAdapter(module.JsonSchema).validate_python(schema) == schema
    assert TypeAdapter(module.Bare).validate_python({"k": 1}) == {"k": 1}
    opened = module.Open.model_validate({"a": "1", "b": 2})
    assert opened.model_dump(by_alias=True) == {"a": "1", "b": 2}
    with pytest.raises(ValidationError, match="b"):
        module.Closed.model_validate({"a": "1", "b": 2})
