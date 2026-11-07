"""Fixtures for importing generated models from reviewed test schemas."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from aiollm_codegen import generate_models

EXAMPLE = Path(__file__).parents[1] / "examples" / "provider.openapi.yaml"


@pytest.fixture
def spec(tmp_path: Path) -> Callable[..., Path]:
    def write(schemas: dict[str, Any], version: str = "3.1.0") -> Path:
        path = tmp_path / "openapi.json"
        path.write_text(
            json.dumps(
                {
                    "openapi": version,
                    "info": {"title": "Test", "version": "1"},
                    "paths": {},
                    "components": {"schemas": schemas},
                }
            ),
            encoding="utf-8",
        )
        return path

    return write


@pytest.fixture
def models(tmp_path: Path) -> Iterator[Callable[..., Any]]:
    loaded: list[str] = []

    def load(source: Path = EXAMPLE, **options: Any) -> Any:
        name = f"test_generated_{len(loaded)}"
        path = tmp_path / f"{name}.py"
        path.write_text(generate_models(source, **options), encoding="utf-8")
        module_spec = importlib.util.spec_from_file_location(name, path)
        assert module_spec is not None and module_spec.loader is not None
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[name] = module
        loaded.append(name)
        module_spec.loader.exec_module(module)
        return module

    yield load
    for name in loaded:
        sys.modules.pop(name)
