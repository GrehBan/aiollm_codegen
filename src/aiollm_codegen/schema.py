"""Load bundled OpenAPI component schemas without resolving external files."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Final
from urllib.parse import unquote

import yaml
from pydantic import JsonValue, TypeAdapter

from aiollm_codegen.normalization import json_schema_document, schema_nodes

__all__ = [
    "load_document",
    "load_schemas",
    "merge_documents",
    "schema_nodes",
    "select_schemas",
]

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


def merge_documents(
    base: dict[str, JsonValue], *others: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    """Add the paths and components of other documents to one.

    One package can then come from several specifications, such as a
    provider's contract snapshot and a document of its other endpoints;
    the base keeps its ``info`` and ``servers``. Every component section
    (``schemas``, ``parameters``, ``requestBodies``, ...) is merged, so a
    merged operation keeps what it refers to; a path's own fields, such
    as its ``parameters``, may appear in both documents when they agree.

    Raises:
        ValueError: If two documents declare the same operation, or the
            same component or path field differently.
    """
    merged = dict(base)
    paths = dict(_object(merged.get("paths", {}), "paths"))
    components = dict(_object(merged["components"], "components"))
    for other in others:
        for path, item in _object(other.get("paths", {}), "paths").items():
            fields = dict(_object(paths.get(path, {}), f"path {path}"))
            for key, value in _object(item, f"path {path}").items():
                if key in fields and key in HTTP_METHODS:
                    raise ValueError(
                        f"Operation declared twice: {key.upper()} {path}"
                    )
                if _differ(fields.setdefault(key, value), value):
                    raise ValueError(
                        f"Documents disagree on {key} of path {path}"
                    )
            paths[path] = fields
        extra = _object(other.get("components", {}), "components")
        for section, entries in extra.items():
            label = f"components.{section}"
            known = dict(_object(components.get(section, {}), label))
            for name, value in _object(entries, label).items():
                if _differ(known.setdefault(name, value), value):
                    kind = "schema" if section == "schemas" else section
                    raise ValueError(f"Documents disagree on {kind} {name}")
            components[section] = known
    merged["paths"] = paths
    merged["components"] = components
    return merged


HTTP_METHODS: Final = frozenset(
    {"get", "put", "post", "delete", "options", "head", "patch", "trace"}
)
"""The keys of a path item that are operations."""


def _differ(first: JsonValue, second: JsonValue) -> bool:
    """Compare two JSON values by their JSON, so ``true`` is not ``1``."""
    return json.dumps(first, sort_keys=True) != json.dumps(
        second, sort_keys=True
    )


def _object(value: JsonValue, label: str) -> dict[str, JsonValue]:
    """Require an object in a document being merged."""
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object for {label}")
    return value


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
