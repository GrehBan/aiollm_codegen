"""Group generated schemas into logical Python modules."""

from __future__ import annotations

import ast
import keyword
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from graphlib import TopologicalSorter
from typing import Final, Literal
from urllib.parse import unquote

from pydantic import JsonValue

from aiollm_codegen.docstrings import format_docstrings
from aiollm_codegen.generator import GenerationMetadata, generate_document

_FALLBACK_GROUP: Final = "shared"


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
    if not result or result[0].isdigit() or keyword.iskeyword(result):
        raise ValueError(f"Cannot create a Python class name from {value!r}")
    return result


def module_name(value: str) -> str:
    """Normalize a resource tag or explicit group to a safe module name."""
    result = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    result = re.sub(r"[^a-zA-Z0-9]+", "_", result).strip("_").lower()
    if (
        not result.isidentifier()
        or keyword.iskeyword(result)
        or result == "init"
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
        (
            group
            for group, terms in families
            if any(t in lowered for t in terms)
        ),
        _FALLBACK_GROUP,
    )


@dataclass
class _Unit:
    """Definitions that must share a module because they reference each other.

    Attributes:
        names: Definition names in source order.
        group: Semantic family requested for the unit.
        dependencies: Other units referenced by the definitions.
    """

    names: list[str]
    group: str
    dependencies: set[int] = field(default_factory=set)


def _definition_name(statement: ast.stmt) -> str | None:
    """Name a top-level class or type alias emitted by the backend."""
    if isinstance(statement, ast.ClassDef):
        return statement.name
    if isinstance(statement, ast.TypeAlias):
        return statement.name.id
    return None


def _rebuilt_name(statement: ast.stmt) -> str | None:
    """Return the model of a trailing ``Model.model_rebuild()`` call."""
    if (
        isinstance(statement, ast.Expr)
        and isinstance(call := statement.value, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "model_rebuild"
        and isinstance(call.func.value, ast.Name)
    ):
        return call.func.value.id
    return None


def _used_names(nodes: Iterable[ast.AST]) -> set[str]:
    """Collect every name loaded by the given statements."""
    return {
        node.id
        for statement in nodes
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
    }


def _reachable(graph: Mapping[str, set[str]], start: str) -> set[str]:
    """Return every definition that start refers to, directly or not."""
    seen: set[str] = set()
    pending = [start]
    while pending:
        for following in graph[pending.pop()] - seen:
            seen.add(following)
            pending.append(following)
    return seen


def _components(graph: Mapping[str, set[str]]) -> list[list[str]]:
    """Group mutually recursive definitions, dependencies first."""
    reach = {name: _reachable(graph, name) for name in graph}
    located: dict[str, int] = {}
    components: list[list[str]] = []
    for name in graph:
        if name not in located:
            component = [
                other
                for other in graph
                if other == name
                or (other in reach[name] and name in reach[other])
            ]
            located.update(dict.fromkeys(component, len(components)))
            components.append(component)
    order = TopologicalSorter(
        {
            number: {
                located[dependency]
                for name in component
                for dependency in graph[name]
            }
            - {number}
            for number, component in enumerate(components)
        }
    )
    return [components[number] for number in order.static_order()]


def _family_order(units: Sequence[_Unit]) -> dict[str, int]:
    """Rank families so that as few references as possible point upward.

    An exact dynamic program over family subsets; the number of families
    is small, and ties are broken by name to keep output deterministic.
    """
    weights: dict[tuple[str, str], int] = defaultdict(int)
    for unit in units:
        for number in unit.dependencies:
            if units[number].group != unit.group:
                weights[unit.group, units[number].group] += 1
    families = sorted({unit.group for unit in units})
    best: dict[frozenset[str], tuple[int, tuple[str, ...]]] = {
        frozenset(): (0, ())
    }
    for _ in families:
        following: dict[frozenset[str], tuple[int, tuple[str, ...]]] = {}
        for placed, (cost, order) in best.items():
            for family in families:
                if family in placed:
                    continue
                candidate = (
                    cost + sum(weights[lower, family] for lower in placed),
                    (*order, family),
                )
                key = placed | {family}
                if key not in following or candidate < following[key]:
                    following[key] = candidate
        best = following
    return {
        family: rank
        for rank, family in enumerate(best[frozenset(families)][1])
    }


def _place(units: Sequence[_Unit]) -> list[str]:
    """Assign units to family modules without creating import cycles.

    Families are ranked so that modules import only lower-ranked modules.
    A unit joins its own family unless something it needs lives in a
    higher-ranked family; then it joins the highest such family. Units
    arrive dependencies first, so every dependency is already placed.
    """
    rank = _family_order(units)
    modules: list[str] = []
    for unit in units:
        modules.append(
            max(
                (
                    unit.group,
                    *(modules[number] for number in unit.dependencies),
                ),
                key=rank.__getitem__,
            )
        )
    return modules


def _owners(metadata: GenerationMetadata) -> dict[str, str]:
    """Map each generated class to the component schema that declared it."""
    return {
        model.class_name: unquote(
            model.source_ref.split("#/components/schemas/", 1)[-1].partition(
                "/"
            )[0]
        )
        .replace("~1", "/")
        .replace("~0", "~")
        for model in metadata.models
    }


def _imports(header: Sequence[ast.stmt], used: set[str]) -> list[ast.stmt]:
    """Keep the header imports that a module actually uses, once each."""
    result: list[ast.stmt] = []
    seen: set[tuple[str | None, str, str | None]] = set()
    for statement in header:
        if not isinstance(statement, ast.ImportFrom):
            continue
        names = [
            alias
            for alias in statement.names
            if (
                statement.module == "__future__"
                or (alias.asname or alias.name) in used
            )
            and (statement.module, alias.name, alias.asname) not in seen
        ]
        seen.update(
            (statement.module, alias.name, alias.asname) for alias in names
        )
        if names:
            result.append(
                ast.ImportFrom(statement.module, names, statement.level)
            )
    return result


def _units(
    definitions: Mapping[str, ast.stmt],
    owners: Mapping[str, str],
    groups: Mapping[str, str],
) -> list[_Unit]:
    """Group recursive definitions and order the groups dependencies first."""
    graph = {
        name: (_used_names([statement]) & set(definitions)) - {name}
        for name, statement in definitions.items()
    }
    order = {name: number for number, name in enumerate(definitions)}
    components = [
        sorted(component, key=order.__getitem__)
        for component in _components(graph)
    ]
    position = {
        name: number
        for number, component in enumerate(components)
        for name in component
    }
    units = [
        _Unit(
            component,
            module_name(_unit_group(component, owners, groups)),
            {
                position[dependency]
                for name in component
                for dependency in graph[name]
            }
            - {number},
        )
        for number, component in enumerate(components)
    ]
    _adopt_shared(units, owners, groups)
    return units


def _unit_group(
    names: Sequence[str], owners: Mapping[str, str], groups: Mapping[str, str]
) -> str:
    """Use the family of the first definition that has a component owner."""
    return next(
        (
            groups.get(owners[name], logical_group(owners[name]))
            for name in names
            if name in owners
        ),
        _FALLBACK_GROUP,
    )


def _adopt_shared(
    units: Sequence[_Unit],
    owners: Mapping[str, str],
    groups: Mapping[str, str],
) -> None:
    """Move a fallback type to the one family that uses it.

    ``Model`` used only by ``ListModelsResponse`` belongs with models, not
    in shared. Types used by several families, and explicit overrides,
    stay where they are. Users are visited first, so chains follow.
    """
    users: dict[int, set[int]] = defaultdict(set)
    for number, unit in enumerate(units):
        for dependency in unit.dependencies:
            users[dependency].add(number)
    for number in reversed(range(len(units))):
        unit = units[number]
        if unit.group != _FALLBACK_GROUP or any(
            owners.get(name) in groups for name in unit.names
        ):
            continue
        families = {units[user].group for user in users[number]}
        if len(families) == 1:
            unit.group = families.pop()


def split_module(
    source: str,
    metadata: GenerationMetadata,
    groups: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Split one generated module into acyclic snake_case family modules.

    Definitions follow the family of the component that declared them.
    Mutually recursive definitions stay together; a definition whose family
    module would close an import cycle gets a module of its own.

    Args:
        source: A complete module emitted by the backend and adapted.
        metadata: Backend identities of the generated classes.
        groups: Explicit families by component name.

    Returns:
        Module sources by module name, and module names by definition.
    """
    tree = ast.parse(source)
    header: list[ast.stmt] = []
    definitions: dict[str, ast.stmt] = {}
    rebuilds: dict[str, ast.stmt] = {}
    for statement in tree.body:
        if (name := _definition_name(statement)) is not None:
            definitions[name] = statement
        elif (name := _rebuilt_name(statement)) is not None:
            rebuilds[name] = statement
        elif isinstance(statement, ast.ImportFrom):
            header.append(statement)
    units = _units(definitions, _owners(metadata), groups)
    placement = _place(units)
    located = {
        name: module
        for unit, module in zip(units, placement, strict=True)
        for name in unit.names
    }
    modules: dict[str, str] = {}
    for module in sorted(set(placement)):
        names = [name for name in definitions if located[name] == module]
        body = [definitions[name] for name in names]
        rebuilt = [rebuilds[name] for name in names if name in rebuilds]
        used = _used_names([*body, *rebuilt])
        siblings: dict[str, list[str]] = defaultdict(list)
        for name in sorted(used & set(definitions)):
            if located[name] != module:
                siblings[located[name]].append(name)
        module_tree = ast.Module(
            [
                ast.Expr(ast.Constant(_module_doc(module))),
                *_imports(header, used),
                *(
                    ast.ImportFrom(
                        sibling, [ast.alias(name) for name in imported], 1
                    )
                    for sibling, imported in sorted(siblings.items())
                ),
                *body,
                *rebuilt,
                *ast.parse(f"__all__ = {sorted(names)!r}").body,
            ],
            [],
        )
        ast.fix_missing_locations(module_tree)
        modules[module] = format_docstrings(
            "# Generated by aiollm-codegen. Do not edit.\n"
            + ast.unparse(module_tree)
            + "\n"
        )
    return modules, located


def _module_doc(module: str) -> str:
    """Describe a generated module by its content."""
    return f"Provider {module.replace('_', ' ')} types generated from OpenAPI."


def grouped_models(
    document: dict[str, JsonValue],
    *,
    kind: Literal["request", "response"],
    groups: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, tuple[str, str]], GenerationMetadata]:
    """Generate family modules and an import map for one model profile.

    The backend compiles the profile once, as a single module, so class
    names stay unique and contextual; the module is then split by its
    references instead of by the backend's per-name module layout.

    Raises:
        ValueError: If the profile cannot be generated or laid out.
    """
    code, metadata = generate_document(document, kind=kind)
    modules, located = split_module(code, metadata, groups)
    metadata = GenerationMetadata(
        models=[
            model.model_copy(update={"module": located[model.class_name]})
            for model in metadata.models
            if model.class_name in located
        ]
    )
    imports: dict[str, tuple[str, str]] = {}
    for model in metadata.models:
        pointer = model.source_ref.split("#/components/schemas/", 1)[-1]
        if "/" not in pointer:
            original = unquote(pointer).replace("~1", "/").replace("~0", "~")
            imports[original] = (model.module, model.class_name)
    files = {f"{module}.py": code for module, code in modules.items()}
    files["__init__.py"] = (
        '"""Generated provider component types."""\n'
        + "\n".join(
            f"from .{module} import {symbol} as {symbol}"
            for module, symbol in sorted(
                imports.values(), key=lambda item: item[1]
            )
        )
        + "\n\n__all__ = "
        + repr(sorted(symbol for _, symbol in imports.values()))
        + "\n"
    )
    return files, imports, metadata
