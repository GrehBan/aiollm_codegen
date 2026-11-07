"""Normalize bundled schema representations without changing wire values."""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Final
from urllib.parse import unquote

from pydantic import JsonValue

MAX_SCHEMA_NODES: Final = 100_000
SCHEMA_PREFIX: Final = "#/components/schemas/"
_URL_FORMATS: Final = frozenset({"uri", "url", "uri-reference", "iri"})


def schema_nodes(
    schema: dict[str, JsonValue],
) -> Iterator[dict[str, JsonValue]]:
    """Walk schema keywords, preserving arbitrary examples and enum values."""
    pending = [schema]
    count = 0
    while pending:
        node = pending.pop()
        count += 1
        if count > MAX_SCHEMA_NODES:
            raise ValueError("Schema exceeds the 100,000-node limit")
        yield node
        for key in (
            "properties",
            "patternProperties",
            "$defs",
            "definitions",
            "dependentSchemas",
        ):
            if isinstance(children := node.get(key), dict):
                pending.extend(
                    v for v in children.values() if isinstance(v, dict)
                )
        pending.extend(
            child
            for key in (
                "items",
                "additionalProperties",
                "not",
                "contains",
                "propertyNames",
                "if",
                "then",
                "else",
                "unevaluatedProperties",
                "unevaluatedItems",
                "additionalItems",
            )
            if isinstance(child := node.get(key), dict)
        )
        for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
            if isinstance(children := node.get(key), list):
                pending.extend(v for v in children if isinstance(v, dict))


def normalize_description(text: str) -> str:
    """Decode documented newlines and trim outer blank space."""
    return inspect.cleandoc(
        text.replace("\\r\\n", "\n").replace("\\n", "\n")
    ).strip()


def json_schema_document(
    schema: dict[str, JsonValue], source: Path
) -> dict[str, JsonValue]:
    """Bundle a JSON Schema root and definitions as OpenAPI components."""
    definitions = schema.pop("$defs", schema.pop("definitions", {}))
    if not isinstance(definitions, dict):
        raise ValueError("JSON Schema definitions must be an object")
    title = schema.get("title")
    name = (
        title
        if isinstance(title, str)
        else source.stem.split(".")[0].capitalize()
    )
    if name in definitions and _without_labels(
        definitions[name]
    ) != _without_labels(schema):
        raise ValueError(f"JSON Schema root conflicts with definition: {name}")
    definitions[name] = schema
    for definition in definitions.values():
        if not isinstance(definition, dict):
            raise ValueError("JSON Schema definitions must be object schemas")
        _rewrite_json_references(definition, name)
    return {
        "openapi": "3.1.0",
        "info": {"title": name, "version": "1"},
        "paths": {},
        "components": {"schemas": definitions},
    }


def _without_labels(schema: JsonValue) -> JsonValue:
    """Compare a root with its definition apart from title and dialect."""
    if not isinstance(schema, dict):
        return schema
    return {
        key: value
        for key, value in schema.items()
        if key not in ("title", "$schema")
    }


def _rewrite_json_references(
    definition: dict[str, JsonValue], name: str
) -> None:
    """Translate JSON Schema pointers into bundled component pointers."""
    for node in schema_nodes(definition):
        if isinstance(ref := node.get("$ref"), str):
            if ref == "#":
                node["$ref"] = SCHEMA_PREFIX + name
            for prefix in ("#/$defs/", "#/definitions/"):
                if ref.startswith(prefix):
                    node["$ref"] = SCHEMA_PREFIX + ref[len(prefix) :]
        if isinstance(
            discriminator := node.get("discriminator"), dict
        ) and isinstance(mapping := discriminator.get("mapping"), dict):
            for tag, target in mapping.items():
                if isinstance(target, str) and target.startswith("#/$defs/"):
                    mapping[tag] = SCHEMA_PREFIX + target[len("#/$defs/") :]


def _target(schemas: dict[str, JsonValue], ref: str) -> dict[str, JsonValue]:
    """Resolve a validated bundled schema pointer."""
    value: JsonValue = schemas
    for part in unquote(ref.removeprefix(SCHEMA_PREFIX)).split("/"):
        if not isinstance(value, dict):
            raise ValueError(f"Invalid schema reference: {ref}")
        value = value.get(part.replace("~1", "/").replace("~0", "~"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid schema reference: {ref}")
    return value


def _expand_allof(
    node: dict[str, JsonValue],
    schemas: dict[str, JsonValue],
    active: frozenset[str],
) -> None:
    """Materialize composed object references without mutable inheritance."""
    members = node.get("allOf")
    if not isinstance(members, list):
        return
    expanded: list[JsonValue] = []
    for member in members:
        if not isinstance(member, dict):
            expanded.append(member)
            continue
        if isinstance(ref := member.get("$ref"), str):
            if ref in active:
                raise ValueError("Cyclic allOf composition in schema")
            parent = deepcopy(_target(schemas, ref))
            _expand_allof(parent, schemas, active | {ref})
            expanded.append(parent)
            sibling = {
                key: value for key, value in member.items() if key != "$ref"
            }
            if sibling:
                expanded.append(sibling)
        else:
            _expand_allof(member, schemas, active)
            expanded.append(member)
    node["allOf"] = expanded
    _flatten_object(node, expanded)


def _flatten_object(
    node: dict[str, JsonValue], members: list[JsonValue]
) -> None:
    """Combine object fields while retaining their intersections."""
    if not all(
        isinstance(member, dict)
        and (member.get("type") == "object" or "properties" in member)
        for member in members
    ):
        return
    properties: dict[str, JsonValue] = {}
    required: set[str] = set()
    merged: dict[str, JsonValue] = {}
    for part in (*members, node):
        assert isinstance(part, dict)
        merged.update(
            {
                key: value
                for key, value in part.items()
                if key not in ("allOf", "properties", "required")
            }
        )
        if isinstance(fields := part.get("properties"), dict):
            for name, field in fields.items():
                previous = properties.get(name)
                properties[name] = _intersect_field(previous, field)
        if isinstance(names := part.get("required"), list):
            required.update(name for name in names if isinstance(name, str))
    merged["type"] = "object"
    merged["properties"] = properties
    required_items: list[JsonValue] = []
    required_items.extend(sorted(required))
    merged["required"] = required_items
    node.clear()
    node.update(merged)


def _intersect_field(previous: JsonValue, field: JsonValue) -> JsonValue:
    """Preserve constants when intersecting composed fields."""
    if previous is None or previous == field:
        return field
    if isinstance(previous, dict) and isinstance(field, dict):
        if "const" in field:
            return {**previous, **field}
        if "const" in previous:
            return {**field, **previous}
    return {"allOf": [previous, field]}


def _tag(
    schema: dict[str, JsonValue],
    name: str,
    schemas: dict[str, JsonValue],
    active: frozenset[str] = frozenset(),
) -> str | None:
    """Find a wire discriminator tag, including inherited constants."""
    if isinstance(ref := schema.get("$ref"), str):
        if ref in active:
            return None
        return _tag(_target(schemas, ref), name, schemas, active | {ref})
    if isinstance(properties := schema.get("properties"), dict) and isinstance(
        field := properties.get(name), dict
    ):
        if isinstance(value := field.get("const"), str):
            return value
        if (
            isinstance(enum := field.get("enum"), list)
            and len(enum) == 1
            and isinstance(enum[0], str)
        ):
            return enum[0]
    if isinstance(members := schema.get("allOf"), list):
        for member in members:
            if (
                isinstance(member, dict)
                and (tag := _tag(member, name, schemas, active)) is not None
            ):
                return tag
    return None


def _discriminator(
    node: dict[str, JsonValue], schemas: dict[str, JsonValue]
) -> None:
    """Resolve discriminators using wire constants rather than class names."""
    discriminator = node.get("discriminator")
    variants = node.get("oneOf", node.get("anyOf"))
    if not isinstance(discriminator, dict) or not isinstance(variants, list):
        return
    name = discriminator.get("propertyName")
    if not isinstance(name, str):
        return
    mapping: dict[str, JsonValue] = {}
    for variant in variants:
        if not isinstance(variant, dict) or not isinstance(
            ref := variant.get("$ref"), str
        ):
            node.pop("discriminator")
            return
        tag = _tag(variant, name, schemas)
        if tag is None or tag in mapping:
            node.pop("discriminator")
            return
        mapping[tag] = ref
    discriminator["mapping"] = mapping


def prepare_document(
    document: dict[str, JsonValue], *, request: bool
) -> dict[str, JsonValue]:
    """Normalize documentation, uploads, object composition and wire tags."""
    document = deepcopy(document)
    components = document["components"]
    assert isinstance(components, dict)
    schemas = components["schemas"]
    assert isinstance(schemas, dict)
    for schema in schemas.values():
        assert isinstance(schema, dict)
        for node in schema_nodes(schema):
            _expand_allof(node, schemas, frozenset())
            if isinstance(description := node.get("description"), str):
                node["description"] = normalize_description(description)
            if request and node.get("format") == "binary":
                node["x-python-type"] = "_AiollmInputFile"
            if node.get("format") in _URL_FORMATS and "maxLength" in node:
                # The backend's URL type cannot carry a length limit; a
                # plain string keeps the documented limit instead.
                del node["format"]
    for schema in schemas.values():
        assert isinstance(schema, dict)
        for node in schema_nodes(schema):
            _discriminator(node, schemas)
            if request:
                _close_request_object(node)
            if not request:
                _optional_nullable_fields(node, schemas)
                _open_response_values(node, schemas)
    return document


def _close_request_object(node: dict[str, JsonValue]) -> None:
    """Reject unknown request keys only where the schema lists the keys.

    An object with properties and no ``additionalProperties`` takes no
    others; one without properties (a JSON Schema, metadata) is any
    object, read as ``dict[str, Any]``; an explicit value is kept.
    """
    extra = node.get("additionalProperties", False)
    if node.get("type") != "object" or extra not in (False, True):
        return
    if isinstance(node.get("properties"), dict):
        node.setdefault("additionalProperties", False)
    elif (extra is True or "additionalProperties" not in node) and not any(
        key in node
        for key in ("$ref", "allOf", "anyOf", "oneOf", "patternProperties")
    ):
        node["additionalProperties"] = {}


def _allows_null(
    schema: JsonValue, schemas: dict[str, JsonValue], seen: frozenset[str]
) -> bool:
    """Tell whether a schema accepts null, following component references."""
    if not isinstance(schema, dict):
        return False
    if isinstance(ref := schema.get("$ref"), str):
        if not ref.startswith(SCHEMA_PREFIX) or ref in seen:
            return False
        return _allows_null(_target(schemas, ref), schemas, seen | {ref})
    kind = schema.get("type")
    values = schema.get("enum")
    return (
        kind == "null"
        or (isinstance(kind, list) and "null" in kind)
        or schema.get("nullable") is True
        or (isinstance(values, list) and None in values)
        or any(
            isinstance(members := schema.get(key), list)
            and any(_allows_null(member, schemas, seen) for member in members)
            for key in ("anyOf", "oneOf")
        )
    )


def _optional_nullable_fields(
    node: dict[str, JsonValue], schemas: dict[str, JsonValue]
) -> None:
    """Let a received object leave out a field that may be null.

    Providers mark nullable response fields as required yet omit them
    (Anthropic's ``stop_details``, usage counters); a missing one reads as
    None instead of failing the whole response.
    """
    required = node.get("required")
    fields = node.get("properties")
    if not isinstance(required, list) or not isinstance(fields, dict):
        return
    node["required"] = [
        name
        for name in required
        if not (
            isinstance(name, str)
            and _allows_null(fields.get(name), schemas, frozenset())
        )
    ]


def _open_response_values(
    node: dict[str, JsonValue], schemas: dict[str, JsonValue]
) -> None:
    """Accept future response vocabulary while retaining fixed wire tags."""
    values = node.get("enum")
    if (
        isinstance(values, list)
        and len(values) > 1
        and all(isinstance(value, str) or value is None for value in values)
    ):
        node.pop("enum")
        node["type"] = ["string", "null"] if None in values else "string"
    fields = node.get("properties")
    if not isinstance(fields, dict):
        return
    for name in ("status", "stop_reason", "code", "error_code"):
        field = fields.get(name)
        if not isinstance(field, dict):
            continue
        if isinstance(ref := field.get("$ref"), str):
            field = _target(schemas, ref)
        values = field.get("enum")
        if isinstance(field.get("const"), str) or (
            isinstance(values, list)
            and all(
                isinstance(value, str) or value is None for value in values
            )
        ):
            field.pop("enum", None)
            field.pop("const", None)
            if isinstance(values, list) and None in values:
                nullable_type: list[JsonValue] = ["string", "null"]
                field["type"] = nullable_type
            else:
                field["type"] = "string"
