"""Generated models follow the adjacent aiollm provider snapshots."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml
from aiollm_codegen import generate_models, generate_package
from jsonschema import Draft202012Validator
from pydantic import BaseModel, TypeAdapter

import aiollm

AIOLLM = Path(aiollm.__file__).resolve().parents[2]
CONTRACTS = AIOLLM / "tests/providers/contracts"
PROJECT = Path(__file__).resolve().parents[1]
CASES = ("openai/responses", "anthropic/messages")


def _write_method_package(destination: Path) -> None:
    document = yaml.safe_load(
        (PROJECT / "examples/provider.openapi.yaml").read_text("utf-8")
    )
    schemas = document["components"]["schemas"]
    schemas["CreateRequest"]["properties"]["stream"] = {"type": "boolean"}
    schemas["DeltaEvent"] = {
        "type": "object",
        "required": ["type", "delta"],
        "properties": {
            "type": {"const": "delta"},
            "delta": {"type": "string"},
        },
    }
    schemas["DoneEvent"] = {
        "type": "object",
        "required": ["type"],
        "properties": {"type": {"const": "response.completed"}},
    }
    schemas["ResponseStreamEvent"] = {
        "oneOf": [
            {"$ref": "#/components/schemas/DeltaEvent"},
            {"$ref": "#/components/schemas/DoneEvent"},
        ],
        "discriminator": {"propertyName": "type"},
    }
    document["paths"]["/responses"]["post"]["responses"]["200"]["content"][
        "text/event-stream"
    ] = {"schema": {"$ref": "#/components/schemas/ResponseStreamEvent"}}
    for path, media in (
        ("/events", "text/event-stream"),
        ("/lines", "application/x-ndjson"),
        ("/files/content", "application/octet-stream"),
    ):
        document["paths"][path] = {
            "get": {
                "operationId": path.replace("/", "_").strip("_"),
                "responses": {"200": {"content": {media: {}}}},
            }
        }
    document["paths"]["/files"] = {
        "post": {
            "operationId": "uploadFile",
            "requestBody": {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["file"],
                            "properties": {
                                "file": {"type": "string", "format": "binary"}
                            },
                        }
                    }
                },
            },
            "responses": {"204": {}},
        }
    }
    source = destination / "methods.openapi.json"
    source.write_text(json.dumps(document), encoding="utf-8")
    for relative, code in generate_package(source).items():
        target = destination / "method_package" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code, encoding="utf-8")


@pytest.fixture(scope="module")
def contracts(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[Path, dict[tuple[str, str], tuple[ModuleType, Any]]]]:
    destination = tmp_path_factory.mktemp("contracts")
    modules = {}
    for case in CASES:
        for profile in ("request", "response", "stream"):
            source = CONTRACTS / case / f"{profile}.schema.json"
            name = f"{case.split('/')[0]}_{profile}"
            output = destination / f"{name}.py"
            output.write_text(
                generate_models(
                    source,
                    kind="request" if profile == "request" else "response",
                ),
                encoding="utf-8",
            )
            spec = importlib.util.spec_from_file_location(name, output)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            schema = json.loads(source.read_text("utf-8"))
            modules[case, profile] = module, schema
    _write_method_package(destination)
    yield destination, modules
    for module, _ in modules.values():
        sys.modules.pop(module.__name__)


def test_generated_contracts_validate_all_saved_payloads(
    contracts: tuple[Path, dict[tuple[str, str], tuple[ModuleType, Any]]],
) -> None:
    _, modules = contracts
    count = 0
    for case in CASES:
        for profile, filename in (
            ("response", "responses"),
            ("stream", "events"),
        ):
            module, schema = modules[case, profile]
            adapter = TypeAdapter(
                getattr(module, schema.get("title", "Stream"))
            )
            examples = json.loads(
                (CONTRACTS / case / f"{filename}.json").read_text("utf-8")
            )
            for example in examples:
                value = adapter.validate_python(example["body"])
                assert value.model_dump(mode="json", exclude_unset=True)
                count += 1
            if profile == "stream":
                future = adapter.validate_python(
                    {"type": "future.event", "future_payload": 1}
                )
                assert future.model_extra == {"future_payload": 1}
    assert count == 72


def test_generated_requests_dump_valid_wire_bodies(
    contracts: tuple[Path, dict[tuple[str, str], tuple[ModuleType, Any]]],
) -> None:
    _, modules = contracts
    for case, body in (
        ("openai/responses", {"model": "test", "input": "Hi"}),
        (
            "anthropic/messages",
            {
                "model": "test",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        ),
    ):
        module, schema = modules[case, "request"]
        adapter = TypeAdapter(getattr(module, schema.get("title", "Request")))
        dumped = adapter.dump_python(
            adapter.validate_python(body),
            mode="json",
            exclude_unset=True,
            by_alias=True,
        )
        Draft202012Validator(schema).validate(dumped)
        assert dumped == body


def test_generated_model_identities_belong_to_snapshot(
    contracts: tuple[Path, dict[tuple[str, str], tuple[ModuleType, Any]]],
) -> None:
    _, modules = contracts
    for module, schema in modules.values():
        upstream = set(schema.get("$defs", {})) | {
            schema.get("title", "Request"),
            "Stream",
        }
        for model in vars(module).values():
            if (
                isinstance(model, type)
                and issubclass(model, BaseModel)
                and model.__module__ == module.__name__
            ):
                identity = model.__dict__.get("__upstream_schema__", ())
                assert identity == () or identity in upstream


def test_generated_contracts_pass_actual_aiollm_checkers(
    contracts: tuple[Path, dict[tuple[str, str], tuple[ModuleType, Any]]],
) -> None:
    destination, _ = contracts
    config = str(AIOLLM / "pyproject.toml")
    for args in (
        ["ruff", "check", "--config", config, str(destination)],
        ["ruff", "format", "--check", "--config", config, str(destination)],
        ["mypy", "--config-file", config, "--no-pretty", str(destination)],
        ["pyright", "--project", config, str(destination)],
    ):
        result = subprocess.run(
            [sys.executable, "-m", *args],
            capture_output=True,
            text=True,
            check=False,
            cwd=PROJECT,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_bundled_lint_profile_matches_aiollm() -> None:
    upstream = tomllib.loads((AIOLLM / "pyproject.toml").read_text("utf-8"))[
        "tool"
    ]["ruff"]
    bundled = tomllib.loads(
        (PROJECT / "src/aiollm_codegen/strict.toml").read_text("utf-8")
    )
    assert bundled["lint"] == upstream["lint"]
    assert bundled["line-length"] == upstream["line-length"]
    assert bundled["target-version"] == upstream["target-version"]
