"""CLI generation and failure publication contracts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml
from aiollm_codegen.cli import main
from conftest import EXAMPLE
from datamodel_code_generator import Error


def test_cli_writes_importable_models_and_overwrites_atomically(
    tmp_path: Path,
) -> None:
    target = tmp_path / "nested" / "models.py"
    arguments = [
        str(EXAMPLE),
        "--kind",
        "request",
        "--schema",
        "CreateRequest",
        "-o",
        str(target),
    ]
    assert main(arguments) == 0
    first = target.read_bytes()
    assert b"class CreateRequest" in first
    assert b"class GenerationResponse" not in first
    assert b"\r\n" not in first
    target.write_text("old content", encoding="utf-8")
    assert main(arguments) == 0
    assert target.read_bytes() == first
    assert list(target.parent.glob("*.tmp")) == []
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import runpy; runpy.run_path({str(target)!r})",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_failure_preserves_existing_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "models.py"
    target.write_text("keep me", encoding="utf-8")
    assert (
        main(
            [
                str(tmp_path / "missing.yaml"),
                "--kind",
                "request",
                "-o",
                str(target),
            ]
        )
        == 1
    )
    assert target.read_text("utf-8") == "keep me"
    assert "aiollm-codegen:" in capsys.readouterr().err
    assert main([str(target), "--kind", "request", "-o", str(target)]) == 1
    assert "must not overwrite" in capsys.readouterr().err


@pytest.mark.parametrize("error", [Error("backend"), yaml.YAMLError("yaml")])
def test_cli_reports_known_generation_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], error: Any
) -> None:
    with patch("aiollm_codegen.cli.generate_models", side_effect=error):
        assert (
            main(
                [
                    str(EXAMPLE),
                    "--kind",
                    "response",
                    "-o",
                    str(tmp_path / "models.py"),
                ]
            )
            == 1
        )
    assert str(error) in capsys.readouterr().err


def test_replace_failure_cleans_temporary_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "models.py"
    target.write_text("keep", encoding="utf-8")
    with patch.object(os, "replace", side_effect=PermissionError("locked")):
        assert (
            main(
                [
                    str(EXAMPLE),
                    "--kind",
                    "request",
                    "--schema",
                    "Message",
                    "-o",
                    str(target),
                ]
            )
            == 1
        )
    assert target.read_text("utf-8") == "keep"
    assert list(tmp_path.glob("*.tmp")) == []
    assert "locked" in capsys.readouterr().err


def test_help() -> None:
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
