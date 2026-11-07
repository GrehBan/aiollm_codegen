"""Translate explicit OpenAPI operations into aiollm method plans."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Final
from urllib.parse import unquote

from pydantic import JsonValue

from aiollm_codegen.layout import class_name, module_name, object_value
from aiollm_codegen.normalization import SCHEMA_PREFIX

_STATUS_WIDTH: Final = 3
_VERSION_PREFIX: Final = re.compile(r"^/v[0-9]+(?=/)")
"""An API version segment; aiollm's ``BaseAPI`` adds the version."""


@dataclass(frozen=True)
class Operation:
    """An endpoint whose request and result have explicit supported shapes."""

    name: str
    group: str
    path: str
    verb: str
    params: str
    response: str | None
    response_format: str
    body: str
    query: tuple[str, ...]
    path_fields: tuple[str, ...]
    summary: str
    buffered_stream: bool
    streaming: bool = False
    media: str = ""
    terminal_events: tuple[str, ...] = ()
    error_events: tuple[str, ...] = ()


@dataclass(frozen=True)
class OperationOverrides:
    """Explicit corrections for specifications that omit aiollm details.

    Attributes:
        names: Method class names by operationId.
        streams: Event schemas by operationId; each adds a streamed
            ``text/event-stream`` variant beside the JSON result.
    """

    names: Mapping[str, str] = field(default_factory=dict)
    streams: Mapping[str, str] = field(default_factory=dict)


def _resolve(
    document: dict[str, JsonValue], value: JsonValue
) -> dict[str, JsonValue]:
    """Resolve only local object references, rejecting unresolved cycles."""
    seen: set[str] = set()
    node = object_value(value, "operation object")
    while isinstance(ref := node.get("$ref"), str):
        if not ref.startswith("#/") or ref in seen:
            raise ValueError(f"Unsupported operation reference: {ref}")
        seen.add(ref)
        current: JsonValue = document
        for part in unquote(ref[2:]).split("/"):
            current = object_value(current, "reference").get(
                part.replace("~1", "/").replace("~0", "~"),
            )
        node = object_value(current, ref)
    return node


def _properties(
    document: dict[str, JsonValue],
    schema: JsonValue,
    seen: frozenset[int] = frozenset(),
) -> dict[str, JsonValue]:
    """Read properties of an object schema or object allOf composition."""
    node = _resolve(document, schema)
    if id(node) in seen:
        raise ValueError("Cyclic allOf composition in endpoint schema")
    if isinstance(all_of := node.get("allOf"), list):
        result: dict[str, JsonValue] = {}
        for member in all_of:
            result.update(_properties(document, member, seen | {id(node)}))
        return result
    if node.get("type") != "object" and "properties" not in node:
        raise ValueError(
            "Endpoint JSON bodies and responses must be object schemas"
        )
    return object_value(node.get("properties", {}), "properties")


def _parameter_encoding(
    parameter: dict[str, JsonValue], location: str, name: str
) -> None:
    """Require parameter serialization supported by aiollm methods."""
    style = "simple" if location == "path" else "form"
    explode = location == "query"
    if (
        parameter.get("style", style) != style
        or parameter.get("explode", explode) != explode
        or parameter.get("allowReserved", False)
    ):
        raise ValueError(f"Unsupported parameter serialization: {name}")


def _declared_parameters(
    document: dict[str, JsonValue],
    path: dict[str, JsonValue],
    operation: dict[str, JsonValue],
) -> dict[tuple[str, str], dict[str, JsonValue]]:
    """Collect path and query parameters; operations override paths.

    Optional headers are left to the session (authentication, workspace)
    or to constant method headers; they are not request fields.

    Raises:
        ValueError: For required headers, cookies or unnamed parameters.
    """
    parameters: dict[tuple[str, str], dict[str, JsonValue]] = {}
    for owner in (path, operation):
        values = owner.get("parameters", [])
        if not isinstance(values, list):
            raise ValueError("Operation parameters must be an array")
        for value in values:
            parameter = _resolve(document, value)
            location, name = parameter.get("in"), parameter.get("name")
            if location == "header" and parameter.get("required") is not True:
                continue
            if location not in ("path", "query") or not isinstance(name, str):
                raise ValueError(
                    "Only named path/query parameters are supported"
                )
            parameters[location, name] = parameter
    return parameters


def _parameters(
    document: dict[str, JsonValue],
    path: dict[str, JsonValue],
    operation: dict[str, JsonValue],
) -> tuple[dict[str, JsonValue], list[str], list[str], list[str]]:
    """Collect inherited/overridden simple path and form query parameters."""
    parameters = _declared_parameters(document, path, operation)
    properties: dict[str, JsonValue] = {}
    required: list[str] = []
    query: list[str] = []
    paths: list[str] = []
    for (location, name), parameter in parameters.items():
        _parameter_encoding(parameter, location, name)
        if name in properties:
            raise ValueError(
                f"Parameter name occurs in path and query: {name}"
            )
        schema = object_value(parameter.get("schema"), f"parameter {name}")
        resolved_schema = _resolve(document, schema)
        if resolved_schema.get("type") in ("object", "array") and (
            location == "path" or resolved_schema.get("type") == "object"
        ):
            raise ValueError(f"Unsupported parameter type: {name}")
        if location == "path" and parameter.get("required") is not True:
            raise ValueError(f"Path parameter must be required: {name}")
        properties[name] = schema
        if parameter.get("required") is True:
            required.append(name)
        (paths if location == "path" else query).append(name)
    return properties, required, query, paths


@dataclass(frozen=True)
class _Result:
    """One buffered or streamed representation of an HTTP success body."""

    schema: str | None
    format: str
    streaming: bool = False
    media: str = ""


def _response_content(
    document: dict[str, JsonValue],
    operation: dict[str, JsonValue],
    name: str,
) -> dict[str, JsonValue]:
    """Require one unambiguous set of successful response representations."""
    responses = object_value(operation.get("responses"), "responses")
    successful = [
        _resolve(document, value)
        for status, value in sorted(responses.items())
        if len(status) == _STATUS_WIDTH and status.startswith("2")
    ]
    if not successful:
        raise ValueError(f"No successful response declared for {name}")
    contents = [response.get("content", {}) for response in successful]
    if any(content != contents[0] for content in contents):
        raise ValueError(f"Multiple success response shapes for {name}")
    return object_value(contents[0], "response content")


def _response_model(
    document: dict[str, JsonValue],
    media: JsonValue,
    root: str,
    schemas: dict[str, JsonValue],
    *,
    buffered: bool,
) -> str | None:
    """Retain a component identity or name an inline result/event schema."""
    schema = object_value(media, "response media").get("schema")
    if schema is None and not buffered:
        return None
    if buffered:
        _properties(document, schema)
    resolved = _resolve(document, schema)
    for component, value in schemas.items():
        if value is resolved:
            return component
    _check_component_name(root, schemas)
    schemas[root] = resolved
    return root


def _binary_media(media: str, declaration: JsonValue) -> bool:
    """Recognize download bodies without treating JSON objects as files."""
    node = object_value(declaration, "response media")
    schema = node.get("schema")
    return (
        media == "application/octet-stream"
        or media.startswith(("audio/", "image/", "video/"))
        or (isinstance(schema, dict) and schema.get("format") == "binary")
    )


def _with_stream(
    content: dict[str, JsonValue],
    stream: str,
    schemas: dict[str, JsonValue],
    name: str,
) -> dict[str, JsonValue]:
    """Declare the event stream that a specification leaves out.

    Raises:
        ValueError: If the schema is unknown or the operation already
            streams or has no JSON result to stream beside.
    """
    if stream not in schemas:
        raise ValueError(f"Unknown stream event schema: {stream}")
    if "text/event-stream" in content or "application/json" not in content:
        raise ValueError(
            f"A stream can only be added beside a JSON result: {name}"
        )
    return {
        **content,
        "text/event-stream": {"schema": {"$ref": SCHEMA_PREFIX + stream}},
    }


def _responses(
    document: dict[str, JsonValue],
    operation: dict[str, JsonValue],
    name: str,
    schemas: dict[str, JsonValue],
    stream: str | None,
) -> list[_Result]:
    """Plan buffered JSON, typed events and unbuffered file downloads."""
    content = _response_content(document, operation, name)
    if stream is not None:
        content = _with_stream(content, stream, schemas, name)
    if not content:
        return [_Result(None, "NONE")]
    results: list[_Result] = []
    if "application/json" in content:
        root = _response_model(
            document,
            content["application/json"],
            name + "Result",
            schemas,
            buffered=True,
        )
        results.append(_Result(root, "JSON"))
    elif set(content) == {"text/plain"}:
        return [_Result(None, "TEXT")]
    event_formats = {
        "text/event-stream": "SSE",
        "application/x-ndjson": "NDJSON",
        "application/ndjson": "NDJSON",
    }
    for media, declaration in sorted(content.items()):
        if media in event_formats:
            root = _response_model(
                document, declaration, name + "Event", schemas, buffered=False
            )
            results.append(_Result(root, event_formats[media], True, media))
        elif _binary_media(media, declaration):
            results.append(_Result(None, "BYTES", True, media))
    if not results:
        raise ValueError(f"Unsupported response media types for {name}")
    if sum(result.streaming for result in results) > 1:
        raise ValueError(f"Multiple streaming response media types for {name}")
    return results


def _request_body(
    document: dict[str, JsonValue], operation: dict[str, JsonValue], name: str
) -> tuple[JsonValue, str, dict[str, JsonValue]]:
    """Select aiollm JSON or multipart encoding for a required body."""
    if "requestBody" not in operation:
        return None, "NONE", {}
    request = _resolve(document, operation["requestBody"])
    content = object_value(request.get("content"), "request content")
    formats = {"application/json": "JSON", "multipart/form-data": "MULTIPART"}
    if len(content) != 1 or (media := next(iter(content))) not in formats:
        raise ValueError(
            f"Only JSON or multipart request bodies are supported for {name}"
        )
    if request.get("required") is not True:
        raise ValueError(f"Optional request bodies are unsupported for {name}")
    schema = object_value(content[media], "request media").get("schema")
    return schema, formats[media], _properties(document, schema)


def _operation_group(operation: dict[str, JsonValue], path: str) -> str:
    """Choose the first tag or path resource as the endpoint module."""
    tags = operation.get("tags")
    return module_name(
        tags[0]
        if isinstance(tags, list) and tags and isinstance(tags[0], str)
        else path.strip("/").split("/")[0]
    )


def _check_component_name(name: str, schemas: dict[str, JsonValue]) -> None:
    """Reject a synthetic component that would replace an upstream schema."""
    if name in schemas:
        raise ValueError(f"Synthetic component name already exists: {name}")


def _event_names(
    operation: dict[str, JsonValue], key: str, defaults: tuple[str, ...]
) -> tuple[str, ...]:
    """Validate explicit provider event policy rather than guessing tags."""
    value = operation.get(key)
    if value is None:
        return defaults
    if not isinstance(value, list) or not all(
        isinstance(tag, str) and tag for tag in value
    ):
        raise ValueError(f"{key} must be an array of nonempty event names")
    return tuple(tag for tag in value if isinstance(tag, str))


def _stream_policy(
    operation: dict[str, JsonValue], result: _Result, path: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Match aiollm's known generation protocols or explicit spec overrides."""
    terminal: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    resource = path.rstrip("/").rpartition("/")[2]
    if resource == "responses" and result.schema == "ResponseStreamEvent":
        terminal = (
            "response.completed",
            "response.incomplete",
            "response.failed",
        )
        errors = ("error", "response.failed")
    elif resource == "messages" and result.schema == "MessageStreamEvent":
        terminal = ("message_stop",)
    return (
        _event_names(operation, "x-aiollm-terminal-events", terminal),
        _event_names(operation, "x-aiollm-error-events", errors),
    )


def _stream_name(name: str) -> str:
    """Name a streaming companion consistently with aiollm providers."""
    return "Stream" + name.removeprefix("Create")


def _plan_variant(
    document: dict[str, JsonValue],
    path: str,
    path_item: dict[str, JsonValue],
    verb: str,
    operation: dict[str, JsonValue],
    name: str,
    result: _Result,
    schemas: dict[str, JsonValue],
) -> Operation:
    """Build independent parameters for buffered and streaming variants."""
    properties, required, query, path_fields = _parameters(
        document, path_item, operation
    )
    body, body_format, body_properties = _request_body(
        document, operation, name
    )
    if set(body_properties) & set(properties):
        raise ValueError(f"Body and parameter fields overlap for {name}")
    stream_field = "stream" in body_properties or "stream" in properties
    if stream_field:
        properties["stream"] = {"type": "boolean", "const": result.streaming}
        if result.streaming and "stream" not in required:
            required.append("stream")
    # An upstream schema may already use the plain name, as Anthropic's
    # CreateMessageParams does; the method's own parameters then differ.
    params = next(
        (
            candidate
            for candidate in (name + "Params", name + "MethodParams")
            if candidate not in schemas
        ),
        name + "Params",
    )
    _check_component_name(params, schemas)
    parameter_schema: dict[str, JsonValue] = {
        "type": "object",
        "properties": properties,
        "required": list(required),
    }
    schemas[params] = (
        {"allOf": [deepcopy(body), parameter_schema]}
        if body is not None
        else parameter_schema
    )
    terminal, errors = (
        _stream_policy(operation, result, path)
        if result.format in ("SSE", "NDJSON")
        else ((), ())
    )
    return Operation(
        name,
        _operation_group(operation, path),
        path,
        verb.upper(),
        params,
        result.schema,
        result.format,
        body_format,
        tuple(query),
        tuple(path_fields),
        str(operation.get("description") or operation.get("summary") or name),
        stream_field,
        result.streaming,
        result.media,
        terminal,
        errors,
    )


def _operation_variants(
    document: dict[str, JsonValue],
    path: str,
    path_item: dict[str, JsonValue],
    verb: str,
    operation: dict[str, JsonValue],
    identity: str,
    schemas: dict[str, JsonValue],
    occupied: set[str],
    overrides: OperationOverrides,
) -> list[Operation]:
    """Produce a method per supported request/response representation."""
    name = class_name(overrides.names.get(identity, identity))
    results = _responses(
        document, operation, name, schemas, overrides.streams.get(identity)
    )
    names = [
        _stream_name(name) if result.streaming and len(results) > 1 else name
        for result in results
    ]
    if occupied.intersection(names):
        raise ValueError("Operation names collide after Python normalization")
    return [
        _plan_variant(
            document,
            path,
            path_item,
            verb,
            operation,
            variant_name,
            result,
            schemas,
        )
        for variant_name, result in zip(names, results, strict=True)
    ]


def plan_operations(
    document: dict[str, JsonValue],
    operation_ids: Sequence[str],
    overrides: OperationOverrides | None = None,
) -> list[Operation]:
    """Add request/result schemas for buffered, stream and file methods.

    Raises:
        ValueError: For unsupported operations or overrides that name no
            selected operation.
    """
    overrides = overrides or OperationOverrides()
    schemas = object_value(
        object_value(document["components"], "components")["schemas"],
        "schemas",
    )
    paths = object_value(document.get("paths"), "paths")
    selected: list[Operation] = []
    found: set[str] = set()
    class_names: set[str] = set()
    for path, path_value in sorted(paths.items()):
        path_item = _resolve(document, path_value)
        for verb, value in sorted(path_item.items()):
            if verb not in (
                "get",
                "post",
                "put",
                "patch",
                "delete",
                "head",
                "options",
            ):
                continue
            operation = object_value(value, "operation")
            identity = operation.get("operationId")
            if operation_ids and identity not in operation_ids:
                continue
            if not isinstance(identity, str) or identity in found:
                raise ValueError("Each operation needs a unique operationId")
            found.add(identity)
            variants = _operation_variants(
                document,
                _VERSION_PREFIX.sub("", path),
                path_item,
                verb,
                operation,
                identity,
                schemas,
                class_names,
                overrides,
            )
            for variant in variants:
                class_names.add(variant.name)
                selected.append(variant)
    if unknown := (set(overrides.names) | set(overrides.streams)) - found:
        raise ValueError(
            "Overrides name unselected operations: "
            + ", ".join(sorted(unknown))
        )
    missing = set(operation_ids) - found
    if missing or not selected:
        raise ValueError(
            f"No matching operations: {', '.join(sorted(missing))}"
        )
    return selected
