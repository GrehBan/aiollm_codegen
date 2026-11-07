"""Keep unknown response variants through aiollm's tagged fallback contract."""

from __future__ import annotations

import ast
from collections import defaultdict
from functools import reduce
from typing import Final

_MIN_VARIANTS: Final = 2


def _members(node: ast.expr) -> list[ast.expr]:
    """Flatten a complete union before considering any fallback."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _members(node.left) + _members(node.right)
    return [node]


def _union(members: list[ast.expr]) -> ast.expr:
    """Join nonempty union operands without changing their order."""
    return reduce(
        lambda left, right: ast.BinOp(left, ast.BitOr(), right), members
    )


def _class_tags(node: ast.ClassDef) -> tuple[str, ...]:
    """Read actual type constants from the backend's class annotations."""
    for field in node.body:
        if not (
            isinstance(field, ast.AnnAssign)
            and isinstance(field.target, ast.Name)
            and field.target.id == "type"
        ):
            continue
        for literal in ast.walk(field.annotation):
            if not (
                isinstance(literal, ast.Subscript)
                and isinstance(literal.value, ast.Name)
                and literal.value.id == "Literal"
            ):
                continue
            values = (
                literal.slice.elts
                if isinstance(literal.slice, ast.Tuple)
                else [literal.slice]
            )
            return tuple(
                value.value
                for value in values
                if isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            )
    return ()


class _Fallbacks(ast.NodeTransformer):
    """Use callable discrimination so invalid known variants remain errors."""

    def __init__(
        self, tree: ast.Module, tags: dict[str, tuple[str, ...]]
    ) -> None:
        self.tags = tags
        self.used = False
        occupied = {
            node.name if isinstance(node, ast.ClassDef) else node.name.id
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.TypeAlias))
        }
        self.fallback = "UnknownVariant"
        while self.fallback in occupied:
            self.fallback += "Value"

    def _tags(self, node: ast.expr) -> tuple[str, ...]:
        """Resolve the wire tags of a model referenced by name."""
        if isinstance(node, ast.Name):
            return self.tags.get(node.id, ())
        return ()

    def visit_BinOp(self, node: ast.BinOp) -> ast.expr:
        """Wrap only complete unions whose object variants have wire tags."""
        if not isinstance(node.op, ast.BitOr):
            return node
        variants = _members(node)
        nullable = [
            v
            for v in variants
            if isinstance(v, ast.Constant) and v.value is None
        ]
        variants = [v for v in variants if v not in nullable]
        tags = [self._tags(variant) for variant in variants]
        if len(variants) < _MIN_VARIANTS or not all(tags):
            return node
        grouped: dict[str, list[ast.expr]] = defaultdict(list)
        for variant, values in zip(variants, tags, strict=True):
            for value in values:
                grouped[value].append(variant)
        mapping = {tag: str(index) for index, tag in enumerate(grouped)}
        members = [
            ast.parse(
                f"Annotated[{ast.unparse(_union(values))}, "
                f"Tag({mapping[tag]!r})]",
                mode="eval",
            ).body
            for tag, values in grouped.items()
        ]
        members.append(
            ast.parse(
                f"Annotated[{self.fallback}, Tag(UNKNOWN)]", mode="eval"
            ).body
        )
        result = ast.parse(
            f"Annotated[{ast.unparse(_union(members))}, "
            f"Discriminator(by_type({mapping!r}))]",
            mode="eval",
        ).body
        self.used = True
        return _union([result, *nullable])

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        """Replace field discrimination while retaining other metadata."""
        self.generic_visit(node)
        if not (
            isinstance(node.value, ast.Name)
            and node.value.id == "Annotated"
            and isinstance(node.slice, ast.Tuple)
        ):
            return node
        replaced = any(
            isinstance(part, ast.Call)
            and isinstance(part.func, ast.Name)
            and part.func.id == "by_type"
            for part in ast.walk(node.slice.elts[0])
        )
        if replaced:
            for part in node.slice.elts[1:]:
                if (
                    isinstance(part, ast.Call)
                    and isinstance(part.func, ast.Name)
                    and part.func.id == "Field"
                ):
                    part.keywords = [
                        keyword
                        for keyword in part.keywords
                        if keyword.arg != "discriminator"
                    ]
        return node


def add_unknown_variants(source: str) -> str:
    """Adapt tagged response unions of one generated module."""
    tree = ast.parse(source)
    tags = {
        node.name: values
        for node in tree.body
        if isinstance(node, ast.ClassDef) and (values := _class_tags(node))
    }
    adapter = _Fallbacks(tree, tags)
    adapter.visit(tree)
    if adapter.used:
        additions = ast.parse(
            "from typing import Annotated\n"
            "from pydantic import Discriminator, Tag\n"
            "from aiollm.types.unknown import (\n"
            "    UNKNOWN, BaseUnknown, by_type,\n)\n"
            f"class {adapter.fallback}(BaseUnknown):\n"
            '    """An unknown provider variant retaining every field.\n\n'
            "    Attributes:\n"
            "        type: The provider's original variant tag.\n"
            '    """\n'
            "    __upstream_schema__ = ()\n"
        ).body
        position = next(
            (
                index
                for index, node in enumerate(tree.body)
                if isinstance(node, (ast.ClassDef, ast.TypeAlias))
            ),
            len(tree.body),
        )
        tree.body[position:position] = additions
        ast.fix_missing_locations(tree)
    return (
        "# Generated by aiollm-codegen. Do not edit.\n"
        + ast.unparse(tree)
        + "\n"
    )
