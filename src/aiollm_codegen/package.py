"""Generate importable provider packages with separate types and methods."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Final, Literal

from datamodel_code_generator.format import (
    PythonVersion,
    apply_builtin_formatter,
)
from pydantic import JsonValue

from aiollm_codegen.docstrings import format_docstrings
from aiollm_codegen.formatting import format_source
from aiollm_codegen.generator import GenerationMetadata
from aiollm_codegen.layout import grouped_models, object_value
from aiollm_codegen.operations import (
    Operation,
    OperationOverrides,
    plan_operations,
)
from aiollm_codegen.schema import load_document, select_schemas

_PARAMS: Final = "methods/params"
"""Request types live beside the methods that send them, as in aiollm."""


def _format(code: str) -> str:
    """Apply the same documentation and code formatting as model generation."""
    return apply_builtin_formatter(
        format_docstrings(code),
        line_length=79,
        string_normalization=True,
        python_version=PythonVersion.PY_313,
    )


def _exports(description: str, imports: Mapping[str, str]) -> str:
    """Export public names in a deterministic package initializer."""
    return _format(
        repr(description)
        + "\n"
        + "\n".join(
            f"from .{module} import {name} as {name}"
            for name, module in sorted(imports.items())
        )
        + "\n__all__ = "
        + repr(sorted(imports))
        + "\n"
    )


def _method(
    operation: Operation,
    request_imports: Mapping[str, tuple[str, str]],
    response_imports: Mapping[str, tuple[str, str]],
    metadata: GenerationMetadata,
    headers: Mapping[str, str],
) -> tuple[list[str], str]:
    """Render an aiollm method using backend-resolved field names."""
    request_module, request_name = request_imports[operation.params]
    param_model = next(
        model
        for model in metadata.models
        if (model.module, model.class_name) == (request_module, request_name)
    )
    fields = {
        field.original_name or field.alias: field.name
        for field in param_model.fields
    }
    placeholders = re.findall(r"{([^{}]+)}", operation.path)
    if set(placeholders) != set(operation.path_fields):
        raise ValueError(
            f"Path parameters differ from the template for {operation.name}"
        )
    path = re.sub(
        r"{([^{}]+)}",
        lambda match: "{" + fields[match[1]] + "}",
        operation.path,
    )
    imports = [
        f"from .params.{request_module} "
        f"import {request_name} as _{operation.name}Params"
    ]
    result = {"NONE": "None", "TEXT": "str", "BYTES": "bytes"}.get(
        operation.response_format
    )
    if operation.response is not None:
        response_module, response_name = response_imports[operation.response]
        result = f"_{operation.name}Result"
        imports.append(
            f"from ..types.{response_module} "
            f"import {response_name} as {result}"
        )
    elif operation.response_format == "SSE":
        result = "ServerSentEvent"
        imports.append("from aiollm import ServerSentEvent")
    elif operation.response_format == "NDJSON":
        result = "JsonValue"
        imports.append("from pydantic import JsonValue")
    query = tuple(sorted(fields[name] for name in operation.query))
    body_format = operation.body
    base = "BaseStreamMethod" if operation.streaming else "BaseAPIMethod"
    summary = operation.summary
    if operation.streaming:
        summary = (
            (
                "Stream response bytes without buffering the file."
                if operation.response_format == "BYTES"
                else "Stream response events."
            )
            + "\n\n"
            + summary
        )
        if operation.buffered_stream:
            summary += (
                "\n\nAttributes:\n"
                "    stream: Always True; requests a streamed response."
            )
    source = (
        f"class {operation.name}(_{operation.name}Params, "
        f"{base}[{result}]):\n"
        f"    {summary!r}\n"
        f"    __request_path__ = {path!r}\n"
        f"    __request_method__ = {operation.verb!r}\n"
        f"    __request_body_format__ = BodyFormat.{body_format}\n"
        "    __response_format__ = ResponseFormat."
        f"{operation.response_format}\n"
        f"    __request_query__ = frozenset({query!r})\n"
    )
    if operation.streaming:
        headers = {**headers, "Accept": operation.media}
    if headers:
        source += f"    __request_headers__ = {dict(headers)!r}\n"
    if operation.streaming:
        imports.extend(_stream_imports(operation))
        source += _stream_fields(operation)
    return imports, source


def _stream_imports(operation: Operation) -> list[str]:
    """Import the existing aiollm protocol machinery used by this method."""
    imports: list[str] = []
    if operation.buffered_stream:
        imports.extend(
            [
                "from typing import Annotated, Literal",
                "from pydantic import Field",
            ]
        )
    if operation.terminal_events or operation.error_events:
        imports.append("from aiollm.types.matchers import EventTypes")
    if operation.error_events:
        imports.extend(
            [
                "from aiollm.types.matchers import TopLevelError",
                "from aiollm.response.parse.stream "
                "import StreamResponseParser",
            ]
        )
    return imports


def _stream_fields(operation: Operation) -> str:
    """Declare completion/error policy and stream=True."""
    source = ""
    if operation.terminal_events:
        source += (
            "    __stream_terminal__ = EventTypes"
            f"{operation.terminal_events!r}\n"
        )
    if operation.error_events:
        source += (
            "    __stream_parser__ = StreamResponseParser(\n"
            f"        error_events=EventTypes{operation.error_events!r}"
            " | TopLevelError()\n"
            "    )\n"
        )
    if operation.buffered_stream:
        source += (
            "\n    stream: Annotated[Literal[True], Field(\n"
            "        description='Always true; requests a streamed response.'"
            "\n    )] = True\n"
        )
    return source


def _operation_groups(
    operations: Sequence[Operation], groups: dict[str, str]
) -> None:
    """Keep each operation's parameters and result with its endpoint group.

    A result shared by several operations stays with the first; explicit
    group overrides keep precedence over every result.
    """
    explicit = set(groups)
    for operation in operations:
        groups[operation.params] = operation.group
        result = operation.response
        if result is None or result in explicit:
            continue
        if result == operation.name + "Result":
            groups[result] = operation.group
        elif not operation.streaming:
            # Event unions keep their own family (events), as in aiollm.
            groups.setdefault(result, operation.group)


def generate_package(
    source: Path | dict[str, JsonValue],
    *,
    operation_ids: Sequence[str] = (),
    schema_groups: Mapping[str, str] | None = None,
    overrides: OperationOverrides | None = None,
    headers: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a complete provider package as relative Python paths and source.

    Methods are grouped by the first operation tag (or the URL resource).
    Types are grouped by stable semantic families within request/response
    profiles. Explicit schema group overrides take precedence.
    ``overrides`` names methods and adds event streams a specification
    leaves out; ``headers`` are constant headers sent by every method,
    such as an API version.

    ``source`` is an OpenAPI file or an already loaded document, such as
    one from ``contract_document``.

    Raises:
        ValueError: For unsupported operations, ambiguous contracts or names.
        OSError: If the OpenAPI source cannot be read.
        datamodel_code_generator.Error: If backend generation fails.
    """
    document = (
        load_document(source) if isinstance(source, Path) else deepcopy(source)
    )
    available = object_value(
        object_value(document["components"], "components")["schemas"],
        "schemas",
    )
    groups = dict(schema_groups or {})
    if unknown := set(groups) - set(available):
        raise ValueError(
            f"Unknown schema group overrides: {', '.join(sorted(unknown))}"
        )
    operations = plan_operations(document, operation_ids, overrides)
    _operation_groups(operations, groups)
    files: dict[str, str] = {}
    imports: dict[str, dict[str, tuple[str, str]]] = {}
    request_metadata = GenerationMetadata(models=[])
    kinds: tuple[Literal["request", "response"], ...] = ("request", "response")
    for kind in kinds:
        roots = sorted(
            {
                root
                for operation in operations
                if (
                    root := operation.params
                    if kind == "request"
                    else operation.response
                )
                is not None
            }
        )
        if not roots:
            imports[kind] = {}
            continue
        profile = select_schemas(
            deepcopy(document), request=kind == "request", schemas=roots
        )
        modules, imports[kind], metadata = grouped_models(
            profile, kind=kind, groups=groups
        )
        if kind == "request":
            request_metadata = metadata
        folder = _PARAMS if kind == "request" else "types"
        for path, code in modules.items():
            files[f"{folder}/{path}"] = _format(code)
    method_groups: dict[str, list[str]] = defaultdict(list)
    method_imports: dict[str, set[str]] = defaultdict(set)
    method_exports: dict[str, str] = {}
    for operation in operations:
        dependencies, method = _method(
            operation,
            imports["request"],
            imports["response"],
            request_metadata,
            headers or {},
        )
        method_groups[operation.group].append(method)
        method_imports[operation.group].update(dependencies)
        method_exports[operation.name] = operation.group
    if "params" in method_groups:
        raise ValueError("Method group 'params' collides with methods/params")
    for group, methods in sorted(method_groups.items()):
        files[f"methods/{group}.py"] = _format(
            repr(f"API methods for {group}.") + "\n"
            "from __future__ import annotations\n"
            + "from aiollm import (BaseAPIMethod, BaseStreamMethod, "
            "BodyFormat, ResponseFormat)\n"
            + "\n".join(sorted(method_imports[group]))
            + "\n\n"
            + "\n\n".join(methods)
            + "\n__all__ = "
            + repr(sorted(op.name for op in operations if op.group == group))
            + "\n"
        )
    files["methods/__init__.py"] = _exports(
        "Generated endpoint methods.", method_exports
    )
    files.setdefault(
        "types/__init__.py", '"""Generated provider data types."""\n'
    )
    files["__init__.py"] = _exports(
        "Generated provider API.", dict.fromkeys(method_exports, "methods")
    )
    for path, code in files.items():
        code = format_source(code)
        compile(code, path, "exec")
        files[path] = code
    return files
