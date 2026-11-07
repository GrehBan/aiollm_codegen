"""Render generated documentation strings without changing model values."""

from __future__ import annotations

import ast
import inspect
import io
import re
import textwrap
import tokenize
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import accumulate
from typing import Final

_DOC_WIDTH: Final = 65


@dataclass(frozen=True)
class DocumentationContext:
    """What a class docstring may say about where a type comes from.

    Attributes:
        api: The API's title, such as "OpenAI API", when known.
        upstream: Component names of named types; inline types are absent.
        owners: The component each generated type was declared in.
        descriptions: Upstream prose of generated types, by type name.
    """

    api: str | None = None
    upstream: Mapping[str, str] = field(default_factory=dict)
    owners: Mapping[str, str] = field(default_factory=dict)
    descriptions: Mapping[str, str] = field(default_factory=dict)


def _identity(name: str, context: DocumentationContext) -> str:
    """Say what a type is when upstream gives it no short description."""
    source = f"the {context.api}" if context.api else "the provider schema"
    upstream = context.upstream.get(name)
    owner = context.owners.get(name)
    candidates = (
        [
            f"`{name}` from {source} (upstream: `{upstream}`).",
            f"`{name}` from {source}.",
            f"`{name}` (upstream: `{upstream}`).",
        ]
        if upstream is not None
        else [
            f"`{name}`, part of `{owner}` in {source}.",
            f"`{name}`, part of `{owner}`.",
        ]
        if owner is not None and owner != name
        else [f"`{name}` from {source}."]
    )
    return next(
        (line for line in candidates if len(line) <= _DOC_WIDTH),
        f"`{name}`.",
    )


def _short_summary(text: str) -> bool:
    """Tell whether prose starts with a sentence that fits a summary line."""
    first = " ".join(text.split("\n\n", maxsplit=1)[0].split())
    sentence = re.search(r"[.!?](?:\s|$)", first)
    end = sentence.start() + 1 if sentence else len(first)
    return 0 < end <= _DOC_WIDTH


def class_documentation(
    node: ast.ClassDef, context: DocumentationContext | None = None
) -> str:
    """Describe a model using upstream prose and its actual field metadata.

    A class without a short upstream summary is introduced by name and
    origin; a field is listed only when upstream describes it or the type
    it refers to.
    """
    context = context or DocumentationContext()
    description = ast.get_docstring(node) or ""
    if "Attributes:" in description:
        return description
    if not _short_summary(description):
        identity = _identity(node.name, context)
        description = (
            identity + "\n\n" + description if description else identity
        )
    fields: list[str] = []
    for item in node.body:
        if not (
            isinstance(item, ast.AnnAssign)
            and isinstance(item.target, ast.Name)
            and not item.target.id.startswith("_")
        ):
            continue
        details = _field_description(item, context.descriptions)
        if details is None:
            continue
        fields.append(
            textwrap.fill(
                details,
                width=_DOC_WIDTH,
                initial_indent=f"    {item.target.id}: ",
                subsequent_indent="        ",
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    if fields:
        description += "\n\nAttributes:\n" + "\n".join(fields)
    return description


def _field_description(
    item: ast.AnnAssign, descriptions: Mapping[str, str]
) -> str | None:
    """Use the field's description, else that of the one type it names."""
    for node in ast.walk(item):
        if (
            isinstance(node, ast.keyword)
            and node.arg == "description"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return " ".join(node.value.value.split())
    named = {
        node.id
        for node in ast.walk(item.annotation)
        if isinstance(node, ast.Name) and node.id in descriptions
    }
    if len(named) != 1:
        return None
    paragraph = descriptions[named.pop()].split("\n\n")[0]
    return " ".join(paragraph.split())


def _primary_document(text: str) -> str:
    """Keep a short Google-style summary and wrap the remaining prose."""
    paragraphs = text.split("\n\n")
    first = " ".join(paragraphs[0].splitlines())
    sentence = re.search(r"[.!?](?:\s|$)", first)
    end = sentence.start() + 1 if sentence else len(first)
    summary = first[:end].strip()
    if len(summary) > _DOC_WIDTH:
        # Too long for one line: cut the sentence between words and go on
        # with the rest of it, rather than a placeholder summary.
        cut = first.rfind(" ", 0, _DOC_WIDTH - 3)
        cut = cut if cut > 0 else _DOC_WIDTH - 3
        summary, remainder = first[:cut].rstrip() + "...", first[cut:].strip()
    else:
        remainder = first[end:].strip()
        if summary and summary[-1] not in ".!?":
            summary += "."
    details = ([remainder] if remainder else []) + paragraphs[1:]
    wrapped = [summary]
    for paragraph in details:
        if any(
            line.startswith(("    ", "```", "- ", "* "))
            for line in paragraph.splitlines()
        ):
            wrapped.append(paragraph)
        else:
            wrapped.append(
                textwrap.fill(
                    " ".join(paragraph.splitlines()),
                    width=71,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
    return _wrap_document("\n\n".join(wrapped))


def _wrap_document(text: str) -> str:
    """Wrap long documentation lines while retaining explicit line breaks."""
    result: list[str] = []
    for line in text.split("\n"):
        indent = line[: len(line) - len(line.lstrip())]
        result.extend(
            textwrap.wrap(
                line,
                width=_DOC_WIDTH,
                subsequent_indent=indent,
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [""]
        )
    return "\n".join(result)


def _literal(text: str, column: int) -> str:
    """Quote a docstring with real newlines and matching source indentation."""
    escaped = text.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    if escaped.endswith('"'):
        escaped = escaped[:-1] + '\\"'
    lines = escaped.split("\n")
    indent = " " * column
    content = lines[0] + "".join(
        "\n" + indent + line if line else "\n" for line in lines[1:]
    )
    closing = "\n" + indent if len(lines) > 1 else ""
    return '"""' + content + closing + '"""'


def format_docstrings(code: str) -> str:
    """Normalize and render module, class, field, and alias documentation.

    Only standalone string expressions are documentation. Defaults, enum
    values, regular expressions, and other model data remain untouched.
    """
    tree = ast.parse(code)
    primary = {
        (node.body[0].lineno, node.body[0].col_offset)
        for node in ast.walk(tree)
        if isinstance(
            node,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
    }
    documents = {
        (node.lineno, node.col_offset): inspect.cleandoc(
            node.value.value.replace("\\r\\n", "\n").replace("\\n", "\n")
        ).strip()
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }
    documents = {
        position: _primary_document(text)
        if position in primary and text
        else _wrap_document(text)
        for position, text in documents.items()
    }
    offsets = list(
        accumulate(map(len, code.splitlines(keepends=True)), initial=0)
    )
    replacements: list[tuple[int, int, str]] = []
    for token in tokenize.generate_tokens(io.StringIO(code).readline):
        if token.type == tokenize.STRING and token.start in documents:
            start = offsets[token.start[0] - 1] + token.start[1]
            end = offsets[token.end[0] - 1] + token.end[1]
            replacements.append(
                (start, end, _literal(documents[token.start], token.start[1]))
            )
    for start, end, literal in reversed(replacements):
        code = code[:start] + literal + code[end:]
    return code
