"""Generated types are split into acyclic, content-named modules."""

from __future__ import annotations

import ast
import importlib
import io
import sys
import tokenize
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from aiollm_codegen.formatting import format_source
from aiollm_codegen.generator import GenerationMetadata
from aiollm_codegen.layout import split_module

_HEADER = (
    "from __future__ import annotations\n"
    "from typing import Annotated\n"
    "from aiollm.types.base import BaseAPIType\n"
    "from pydantic import Field\n"
)


def _metadata(*names: str) -> GenerationMetadata:
    return GenerationMetadata.model_validate(
        {
            "models": [
                {
                    "class_name": name,
                    "source_ref": f"#/components/schemas/{name}",
                }
                for name in names
            ]
        }
    )


def _imports(code: str) -> set[str]:
    return {
        node.module or ""
        for node in ast.parse(code).body
        if isinstance(node, ast.ImportFrom) and node.level == 1
    }


@pytest.fixture
def load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[[dict[str, str]], str]]:
    monkeypatch.syspath_prepend(str(tmp_path))
    loaded: list[str] = []

    def write(modules: dict[str, str]) -> str:
        name = f"layout_package_{len(loaded)}"
        package = tmp_path / name
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        for module, code in modules.items():
            (package / f"{module}.py").write_text(code, encoding="utf-8")
        loaded.append(name)
        return name

    yield write
    for name in loaded:
        for module in [m for m in sys.modules if m.startswith(name)]:
            sys.modules.pop(module)


def test_family_cycles_are_resolved_without_internal_modules(
    load: Callable[[dict[str, str]], str],
) -> None:
    source = _HEADER + (
        "class Filter(BaseAPIType):\n"
        "    value: str\n"
        "class SearchTool(BaseAPIType):\n"
        "    filter: Filter\n"
        "class MessageNote(BaseAPIType):\n"
        "    filter: Filter\n"
        "class Item(BaseAPIType):\n"
        "    tool: SearchTool\n"
        "Filter.model_rebuild()\n"
        "SearchTool.model_rebuild()\n"
        "MessageNote.model_rebuild()\n"
        "Item.model_rebuild()\n"
    )
    modules, located = split_module(
        source, _metadata("Filter", "SearchTool", "MessageNote", "Item"), {}
    )
    assert set(modules) == {"messages", "shared", "tools"}
    assert located == {
        "Filter": "shared",
        "SearchTool": "tools",
        "MessageNote": "messages",
        "Item": "tools",
    }
    assert _imports(modules["shared"]) == set()
    assert _imports(modules["tools"]) == {"shared"}
    assert "Field" not in modules["shared"]
    assert "__all__ = ['Item', 'SearchTool']" in modules["tools"]
    package = load(modules)
    tools = importlib.import_module(f"{package}.tools")
    item = tools.Item.model_validate({"tool": {"filter": {"value": "x"}}})
    assert item.tool.filter.value == "x"


def test_recursive_types_share_one_module(
    load: Callable[[dict[str, str]], str],
) -> None:
    source = _HEADER + (
        "class TreeNode(BaseAPIType):\n"
        "    branch: Annotated[Branch | None, Field(default=None)]\n"
        "class Branch(BaseAPIType):\n"
        "    message: MessageNote\n"
        "    child: TreeNode\n"
        "class MessageNote(BaseAPIType):\n"
        "    text: str\n"
        "TreeNode.model_rebuild()\n"
        "Branch.model_rebuild()\n"
        "MessageNote.model_rebuild()\n"
    )
    modules, located = split_module(
        source,
        _metadata("TreeNode", "Branch", "MessageNote"),
        {"Branch": "trees", "TreeNode": "nodes"},
    )
    assert located["TreeNode"] == located["Branch"]
    assert located["MessageNote"] == "messages"
    package = load(modules)
    module = importlib.import_module(f"{package}.{located['TreeNode']}")
    tree = module.TreeNode.model_validate(
        {"branch": {"message": {"text": "a"}, "child": {}}}
    )
    assert tree.branch.child.branch is None


def test_long_strings_are_wrapped_between_words() -> None:
    value = "Specifies the comparison operator to apply. " * 4
    word = "x" * 100
    code = format_source(
        f'"""Wrapped strings."""\nTEXT = {value!r}\nWORD = {word!r}\n'
    )
    assigned = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(code).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
    }
    assert assigned == {"TEXT": value, "WORD": word}
    pieces = [
        ast.literal_eval(token.string)
        for token in tokenize.generate_tokens(io.StringIO(code).readline)
        if token.type == tokenize.STRING
    ]
    text = [piece for piece in pieces[1:] if set(piece) - {"x"}]
    assert "".join(text) == value
    assert len(text) > 1
    assert all(piece.endswith(" ") for piece in text[:-1])


def test_shared_types_join_the_only_family_that_uses_them() -> None:
    source = _HEADER + (
        "class Model(BaseAPIType):\n"
        "    id: str\n"
        "class ListModelsResponse(BaseAPIType):\n"
        "    data: list[Model]\n"
        "class Spare(BaseAPIType):\n"
        "    id: str\n"
        "Model.model_rebuild()\n"
        "ListModelsResponse.model_rebuild()\n"
        "Spare.model_rebuild()\n"
    )
    _, located = split_module(
        source,
        _metadata("Model", "ListModelsResponse", "Spare"),
        {"ListModelsResponse": "models"},
    )
    assert located == {
        "Model": "models",
        "ListModelsResponse": "models",
        "Spare": "shared",
    }
    _, overridden = split_module(
        source,
        _metadata("Model", "ListModelsResponse", "Spare"),
        {"ListModelsResponse": "models", "Model": "shared"},
    )
    assert overridden["Model"] == "shared"
