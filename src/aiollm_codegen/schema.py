"""Load bundled OpenAPI component schemas without resolving external files."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import unquote

import yaml
from pydantic import JsonValue, TypeAdapter

from aiollm_codegen.normalization import json_schema_document, schema_nodes

__all__ = ["load_document", "load_schemas", "schema_nodes", "select_schemas"]

_JSON = TypeAdapter(JsonValue)
_PREFIX = "#/components/schemas/"


class _OpenAPILoader(yaml.SafeLoader):
    """Keep YAML timestamp scalars as their original JSON string values."""


def _timestamp_string(loader: _OpenAPILoader, node: yaml.ScalarNode) -> str:
    """Preserve the scalar's spelling rather than constructing a date."""
    return loader.construct_scalar(node)


_OpenAPILoader.add_constructor(
    "tag:yaml.org,2002:timestamp", _timestamp_string
)


def load_document(
    source: Path, *, allow_json_schema: bool = False
) -> dict[str, JsonValue]:
    """Read a bounded OpenAPI document without resolving references."""
    with source.open("rb") as stream:
        payload = stream.read(32 * 1024 * 1024 + 1)
    if len(payload) > 32 * 1024 * 1024:
        raise ValueError("OpenAPI input exceeds the 32 MiB limit")
    document = _JSON.validate_python(
        json.loads(payload)
        if source.suffix.lower() == ".json"
        else yaml.load(payload, Loader=_OpenAPILoader)
    )
    if (
        isinstance(document, dict)
        and "openapi" not in document
        and allow_json_schema
        and set(document)
        & {
            "$schema",
            "$defs",
            "definitions",
            "type",
            "properties",
            "anyOf",
            "oneOf",
            "allOf",
            "$ref",
        }
    ):
        document = json_schema_document(document, source)
    if not isinstance(document, dict) or not str(
        document.get("openapi", "")
    ).startswith(("3.0.", "3.1.")):
        raise ValueError("Expected an OpenAPI 3.0 or 3.1 document")
    components = document.get("components")
    available = (
        components.get("schemas") if isinstance(components, dict) else None
    )
    if not isinstance(available, dict) or not available:
        raise ValueError("Expected nonempty components.schemas")
    return document


def select_schemas(
    document: dict[str, JsonValue],
    *,
    request: bool,
    schemas: Sequence[str] = (),
) -> dict[str, JsonValue]:
    """Select schemas and their dependencies, removing request defaults.

    Mutates the supplied document, which must be owned by the caller.
    """
    components = document["components"]
    assert isinstance(components, dict)
    available = components["schemas"]
    assert isinstance(available, dict)
    selected: dict[str, JsonValue] = {}
    pending = list(schemas or available)
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        if name not in available:
            raise ValueError(f"Unknown component schema: {name}")
        schema = available[name]
        if not isinstance(schema, dict):
            raise ValueError(f"Component schema must be an object: {name}")
        selected[name] = schema
        for node in schema_nodes(schema):
            if (
                "$dynamicRef" in node
                or "$recursiveRef" in node
                or "$id" in node
            ):
                raise ValueError(
                    "Dynamic references and schema IDs are unsupported"
                )
            if (ref := node.get("$ref")) is not None:
                if not isinstance(ref, str) or not ref.startswith(_PREFIX):
                    raise ValueError(
                        "Bundle references under components.schemas first"
                    )
                pointer = unquote(ref[len(_PREFIX) :]).split("/")
                target = pointer[0].replace("~1", "/").replace("~0", "~")
                pending.append(target)
            if request:
                node.pop("default", None)
    document["components"] = {"schemas": dict(sorted(selected.items()))}
    document["paths"] = {}
    return document


def load_schemas(
    source: Path, *, request: bool, schemas: Sequence[str] = ()
) -> dict[str, JsonValue]:
    """Read JSON/YAML and select bundled component schemas and dependencies.

    Raises:
        ValueError: For unsupported documents, references, or schema names.
        OSError: If the source cannot be read.
    """
    return select_schemas(
        load_document(source, allow_json_schema=True),
        request=request,
        schemas=schemas,
    )
