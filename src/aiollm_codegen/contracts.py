"""Describe one endpoint from aiollm's request/response/stream snapshots."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from pydantic import JsonValue

from aiollm_codegen.layout import object_value
from aiollm_codegen.normalization import SCHEMA_PREFIX
from aiollm_codegen.schema import load_document

_PATH: Final = re.compile(r"^(/[A-Za-z0-9_.-]+)+$")


def contract_document(directory: Path, path: str) -> dict[str, JsonValue]:
    """Build an OpenAPI endpoint from a contract snapshot folder.

    The folder holds ``request.schema.json``, ``response.schema.json`` and
    ``stream.schema.json``, each a JSON Schema with ``$defs``. The endpoint
    is ``POST path``: the request is the body, the response the JSON
    result and the stream union, named ``<Response>StreamEvent``, the
    ``text/event-stream`` items. The operation is ``create<Response>``;
    the API is named after the folders (``anthropic/messages`` reads
    Anthropic Messages API).

    Raises:
        ValueError: For a bad path, an untitled request or response, or a
            definition that differs between the snapshots.
        OSError: If a snapshot cannot be read.
    """
    if not _PATH.fullmatch(path):
        raise ValueError(f"Contract path must be a plain URL path: {path!r}")
    roots: dict[str, str] = {}
    schemas: dict[str, JsonValue] = {}
    for part in ("request", "response", "stream"):
        source = directory / f"{part}.schema.json"
        document = load_document(source, allow_json_schema=True)
        title = object_value(document["info"], "info").get("title")
        components = object_value(
            object_value(document["components"], "components")["schemas"],
            "schemas",
        )
        if part == "stream" and title == "Stream":
            title = roots["response"] + "StreamEvent"
            components[title] = components.pop("Stream")
        if not isinstance(title, str) or title == part.capitalize():
            raise ValueError(f"The {part} snapshot needs a title")
        roots[part] = title
        for name, schema in components.items():
            if schemas.setdefault(name, schema) != schema:
                raise ValueError(f"Snapshots disagree on definition {name}")
    identity = "create" + roots["response"]

    def media(name: str) -> dict[str, JsonValue]:
        return {"schema": {"$ref": SCHEMA_PREFIX + name}}

    return {
        "openapi": "3.1.0",
        # aiollm names snapshot folders <provider>/<endpoint>.
        "info": {
            "title": f"{directory.parent.name.title()} "
            f"{directory.name.title()} API",
            "version": "1",
        },
        "paths": {
            path: {
                "post": {
                    "operationId": identity,
                    "summary": f"Create a {roots['response']}.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": media(roots["request"])
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "OK",
                            "content": {
                                "application/json": media(roots["response"]),
                                "text/event-stream": media(roots["stream"]),
                            },
                        }
                    },
                }
            }
        },
        "components": {"schemas": schemas},
    }
