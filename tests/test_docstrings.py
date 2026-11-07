"""Source rendering preserves documentation structure and string boundaries."""

from __future__ import annotations

import ast
import inspect
import re
from collections.abc import Callable
from pathlib import Path

from aiollm_codegen import generate_models, generate_package
from aiollm_codegen.docstrings import format_docstrings
from conftest import EXAMPLE


def test_multiline_docs_keep_code_indentation_quotes_and_unicode() -> None:
    description = (
        '\n  Summary with "quotes", """triple quotes""", and café.\\n\\n'
        '  Example:\\n      print("hello")\\n\n  '
    )
    data = r"keep\nthis\path"
    code = (
        "class Documented:\n"
        f"    {description!r}\n"
        f"    value = {data!r}\n"
        f"    {'  Field details.\\r\\n  Second line.  '!r}\n"
    )
    rendered = format_docstrings(code)
    tree = ast.parse(rendered)
    model = tree.body[0]
    assert isinstance(model, ast.ClassDef)
    assert ast.get_docstring(model) == (
        'Summary with "quotes", """triple quotes""", and café.\n\n'
        'Example:\n    print("hello")'
    )
    assert len(model.body) == 3
    assignment = model.body[1]
    assert isinstance(assignment, ast.Assign)
    assert isinstance(assignment.value, ast.Constant)
    assert assignment.value.value == data
    field = model.body[2]
    assert isinstance(field, ast.Expr)
    assert isinstance(field.value, ast.Constant)
    assert isinstance(field.value.value, str)
    assert (
        inspect.cleandoc(field.value.value) == "Field details.\nSecond line."
    )
    assert '\n        print("hello")\n    """' in rendered


def test_empty_documentation_loses_blank_edges() -> None:
    rendered = format_docstrings(repr("\n  \n\t\n") + "\n")
    assert rendered == '""""""\n'
    assert ast.get_docstring(ast.parse(rendered)) == ""


_BOILERPLATE = (
    re.compile(r"^\s*[A-Z][\w ]* data\.$", re.MULTILINE),
    re.compile(r"Provider schema\."),
    re.compile(r"^\s+(\w+): (\w[\w ]*)\.$", re.MULTILINE),
)


def _docstrings(code: str) -> list[str]:
    return [
        doc
        for node in ast.walk(ast.parse(code))
        if isinstance(node, (ast.Module, ast.ClassDef))
        and (doc := ast.get_docstring(node))
    ]


def _boilerplate(doc: str) -> list[str]:
    found = [m.group(0) for m in _BOILERPLATE[0].finditer(doc)]
    found += [m.group(0) for m in _BOILERPLATE[1].finditer(doc)]
    found += [
        m.group(0)
        for m in _BOILERPLATE[2].finditer(doc)
        if m.group(2).lower() == m.group(1).strip("_").replace("_", " ")
    ]
    return found


def test_generated_documentation_has_no_placeholders(
    spec: Callable[..., Path],
) -> None:
    source = spec(
        {
            "Tool": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "schema": {"$ref": "#/components/schemas/InputSchema"},
                    "async": {"type": "boolean"},
                    "long": {
                        "type": "object",
                        "description": "A very long first sentence that "
                        "goes on and on past the summary width limit of "
                        "the docstring formatter without a stop",
                        "properties": {"x": {"type": "string"}},
                    },
                },
            },
            "InputSchema": {
                "type": "object",
                "description": "The JSON Schema of the tool input.",
                "properties": {"type": {"type": "string"}},
            },
        }
    )
    for kind in ("request", "response"):
        code = generate_models(source, kind=kind)
        docs = _docstrings(code)
        assert not [b for doc in docs for b in _boilerplate(doc)], docs
        tool = next(d for d in docs if d.startswith("`Tool`"))
        assert "upstream: `Tool`" in tool
        assert "schema_: The JSON Schema of the tool input." in tool
        assert "async_" not in tool and "name:" not in tool
    for example in ("request", "response"):
        code = generate_models(EXAMPLE, kind=example)
        assert not [b for doc in _docstrings(code) for b in _boilerplate(doc)]


def test_generated_package_documentation_has_no_placeholders() -> None:
    files = generate_package(EXAMPLE)
    found = {
        path: problems
        for path, code in files.items()
        if (
            problems := [b for d in _docstrings(code) for b in _boilerplate(d)]
        )
    }
    assert not found
