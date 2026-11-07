"""Command-line entry point for local provider schema generation."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import yaml
from datamodel_code_generator import Error
from pydantic import BaseModel

from aiollm_codegen.contracts import contract_document
from aiollm_codegen.generator import generate_models
from aiollm_codegen.operations import OperationOverrides
from aiollm_codegen.package import generate_package
from aiollm_codegen.publication import publish_package


class _Arguments(BaseModel):
    source: Path
    output: Path
    kind: Literal["request", "response"] | None
    schemas: list[str]
    package: bool
    operations: list[str]
    groups: list[str]
    names: list[str]
    streams: list[str]
    headers: list[str]
    contract: str | None


def _arguments(argv: Sequence[str] | None) -> _Arguments:
    """Parse and validate file/package command-line modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Bundled OpenAPI JSON/YAML")
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--kind", choices=("request", "response"))
    parser.add_argument(
        "--schema",
        dest="schemas",
        action="append",
        default=[],
        help="Component name (repeatable)",
    )
    parser.add_argument(
        "--package",
        action="store_true",
        help="Generate types/ and methods/ directories",
    )
    parser.add_argument(
        "--operation",
        dest="operations",
        action="append",
        default=[],
        help="OpenAPI operationId (repeatable)",
    )
    parser.add_argument(
        "--group",
        dest="groups",
        action="append",
        default=[],
        help="SchemaName=module override (repeatable)",
    )
    parser.add_argument(
        "--contract",
        metavar="PATH",
        help="Read a snapshot folder (request/response/stream.schema.json) "
        "as POST PATH; requires --package",
    )
    for flag, destination, meaning in (
        ("--name", "names", "operationId=MethodClass name"),
        ("--stream", "streams", "operationId=EventSchema stream to add"),
        ("--header", "headers", "Header=value sent by every method"),
    ):
        parser.add_argument(
            flag,
            dest=destination,
            action="append",
            default=[],
            help=f"{meaning} (repeatable)",
        )
    arguments = _Arguments.model_validate(vars(parser.parse_args(argv)))
    if arguments.package:
        if arguments.kind is not None or arguments.schemas:
            parser.error("--package uses --operation, not --kind/--schema")
    elif arguments.kind is None or any(
        (
            arguments.operations,
            arguments.groups,
            arguments.names,
            arguments.streams,
            arguments.headers,
            arguments.contract,
        )
    ):
        parser.error(
            "Use --kind for a file, or --package with --operation/--group"
        )
    return arguments


def _pairs(flag: str, values: list[str]) -> dict[str, str]:
    """Parse repeated ``key=value`` options with distinct keys.

    Raises:
        ValueError: For a value without both sides or a repeated key.
    """
    pairs: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key or not item or key in pairs:
            raise ValueError(f"{flag} requires distinct key=value pairs")
        pairs[key] = item
    return pairs


def main(argv: Sequence[str] | None = None) -> int:
    """Generate models, publishing the output atomically after success."""
    arguments = _arguments(argv)
    temporary: Path | None = None
    try:
        if arguments.source.resolve() == arguments.output.resolve():
            raise ValueError("Output must not overwrite the OpenAPI source")
        if arguments.package:
            if arguments.source.resolve().is_relative_to(
                arguments.output.resolve()
            ):
                raise ValueError(
                    "OpenAPI source must be outside the output package"
                )
            files = generate_package(
                arguments.source
                if arguments.contract is None
                else contract_document(arguments.source, arguments.contract),
                operation_ids=arguments.operations,
                schema_groups=_pairs("--group", arguments.groups),
                overrides=OperationOverrides(
                    names=_pairs("--name", arguments.names),
                    streams=_pairs("--stream", arguments.streams),
                ),
                headers=_pairs("--header", arguments.headers),
            )
            publish_package(files, arguments.output)
            return 0
        assert arguments.kind is not None
        code = generate_models(
            arguments.source, kind=arguments.kind, schemas=arguments.schemas
        )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=arguments.output.parent,
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(code)
        os.replace(temporary, arguments.output)
    except (OSError, ValueError, Error, yaml.YAMLError) as error:
        sys.stderr.write(f"aiollm-codegen: {error}\n")
        return 1
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return 0
