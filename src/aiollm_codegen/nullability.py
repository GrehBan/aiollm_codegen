"""Show null on fields whose type alias accepts it."""

from __future__ import annotations

import ast
from functools import reduce


def _members(node: ast.expr) -> list[ast.expr]:
    """Flatten a ``|`` union into its operands."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _members(node.left) + _members(node.right)
    return [node]


def _is_none(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _union(members: list[ast.expr]) -> ast.expr:
    """Join union operands, keeping their order."""
    return reduce(
        lambda left, right: ast.BinOp(left, ast.BitOr(), right), members
    )


class _Nullable(ast.NodeTransformer):
    """Add ``| None`` to every reference of the given aliases."""

    def __init__(self, names: set[str]) -> None:
        self.names = names

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if node.id in self.names and isinstance(node.ctx, ast.Load):
            return ast.BinOp(node, ast.BitOr(), ast.Constant(None))
        return node


class _SingleNone(ast.NodeTransformer):
    """Keep one ``None`` per union, after the other members."""

    def visit_BinOp(self, node: ast.BinOp) -> ast.expr:
        self.generic_visit(node)
        if not isinstance(node.op, ast.BitOr):
            return node
        members = _members(node)
        others = [member for member in members if not _is_none(member)]
        if len(members) - len(others) <= 1:
            return node
        first = next(
            index for index, member in enumerate(members) if _is_none(member)
        )
        others.insert(min(first, len(others)), ast.Constant(None))
        return _union(others)


def show_null(tree: ast.Module) -> None:
    """Repeat an alias's ``None`` at every place that uses the alias.

    With ``type Metadata = dict[str, str] | None``, a field reads
    ``Metadata | None`` rather than ``Metadata``. Validation and the alias
    itself are unchanged; the field annotation now says that it accepts
    null, as aiollm's hand-written models do. An alias that becomes
    nullable this way is handled in turn.
    """
    marked: set[str] = set()
    while True:
        nullable = {
            statement.name.id
            for statement in tree.body
            if isinstance(statement, ast.TypeAlias)
            and any(_is_none(member) for member in _members(statement.value))
        }
        if not (added := nullable - marked):
            break
        marked |= added
        _Nullable(added).visit(tree)
    _SingleNone().visit(tree)
