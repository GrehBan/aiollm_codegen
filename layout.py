"""Assign schema families to Python modules using backend reference handling."""

from __future__ import annotations

import ast
import keyword
import re
from collections.abc import Mapping
from pathlib import PurePosixPath
from urllib.parse import unquote

from pydantic import JsonValue

from aiollm_codegen.generator import _Metadata, generate_document
from aiollm_codegen.schema import _nodes


def object_value(value: JsonValue, label: str) -> dict[str, JsonValue]:
    """Require an object at an OpenAPI boundary."""
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object for {label}")
    return value


def class_name(value: str) -> str:
    """Convert an operation or schema name into a safe class identifier."""
    result = "".join(
        word[:1].upper() + word[1:]
        for word in re.split(r"[^a-zA-Z0-9]+", value)
    )
    if not result or result[0].isdigit():
        raise ValueError(f"Cannot create a Python class name from {value!r}")
    return result


def module_name(value: str) -> str:
    """Normalize a resource tag or explicit group to a safe module name."""
    result = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    result = re.sub(r"[^a-zA-Z0-9]+", "_", result).strip("_").lower()
    if (
        not result.isidentifier() or keyword.iskeyword(result)
        or result.startswith("_") or result == "init"
    ):
        raise ValueError(f"Invalid Python module group: {value!r}")
    return result


def logical_group(name: str) -> str:
    """Use stable semantic families, with shared as an explicit fallback."""
    lowered = name.lower()
    families = (
        ("events", ("event",)),
        ("usage", ("usage", "tokendetails")),
        ("tools", ("tool", "mcp", "computeraction", "shellaction")),
        ("messages", ("message", "content", "refusal")),
        ("responses", ("response",)),
        ("files", ("file", "upload")),
        ("images", ("image",)),
        ("audio", ("audio", "speech", "voice", "transcription")),
    )
    return next(
        (group for group, terms in families if any(t in lowered for t in terms)),
        "shared",
    )


def _reference(ref: str, renamed: Mapping[str, str]) -> str:
    """Rewrite a bundled component pointer without altering nested pointers."""
    prefix = "#/components/schemas/"
    tail = ref.removeprefix(prefix)
    first, separator, rest = tail.partition("/")
    name = unquote(first).replace("~1", "/").replace("~0", "~")
    return prefix + renamed[name] + separator + rest


def grouped_models(
    document: dict[str, JsonValue],
    *,
    kind: str,
    groups: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, tuple[str, str]], _Metadata]:
    """Generate logical modules and a component-to-symbol import map.

    The backend owns cross-module imports, recursive references and model
    rebuilding. Grouping changes component keys only in the owned document.
    """
    components = object_value(document["components"], "components")
    schemas = object_value(components["schemas"], "schemas")
    renamed = {
        name: f"{module_name(groups.get(name, logical_group(name)))}.{class_name(name)}"
        for name in schemas
    }
    if len(set(renamed.values())) != len(renamed):
        raise ValueError("Component names collide after Python normalization")
    for schema in schemas.values():
        for node in _nodes(object_value(schema, "schema")):
            if isinstance(ref := node.get("$ref"), str):
                node["$ref"] = _reference(ref, renamed)
            if isinstance(discriminator := node.get("discriminator"), dict):
                if isinstance(mapping := discriminator.get("mapping"), dict):
                    for tag, target in mapping.items():
                        if isinstance(target, str):
                            mapping[tag] = (
                                _reference(target, renamed)
                                if target.startswith("#/components/schemas/")
                                else renamed.get(target, target)
                            )
    components["schemas"] = {
        renamed[name]: value for name, value in schemas.items()
    }
    code, metadata = generate_document(
        document, kind="request" if kind == "request" else "response",
        modular=True,
    )
    if isinstance(code, str):
        raise ValueError("Expected grouped Python modules")
    # Restore the original provider identity after namespacing schema keys.
    originals = {new: old for old, new in renamed.items()}
    imports: dict[str, tuple[str, str]] = {}
    for model in metadata.models:
        pointer = model.source_ref.split("#/components/schemas/", 1)[-1]
        root, separator, _ = pointer.partition("/")
        original = originals[root]
        if not separator:
            imports[original] = (model.module, model.class_name)
    files: dict[str, str] = {}
    for parts, content in code.items():
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__upstream_schema__"
                for target in node.targets
            ):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    node.value.value = originals.get(node.value.value, node.value.value)
        from aiollm_codegen.docstrings import format_docstrings

        files[str(PurePosixPath(*parts))] = format_docstrings(ast.unparse(tree) + "\n")
    exports = [
        f"from .{module} import {symbol} as {symbol}"
        for module, symbol in sorted(imports.values(), key=lambda item: item[1])
    ]
    files["__init__.py"] = (
        '"""Generated provider component types."""\n'
        + "\n".join(exports)
        + "\n\n__all__ = " + repr(sorted(symbol for _, symbol in imports.values()))
        + "\n"
    )
    return files, imports, metadata
