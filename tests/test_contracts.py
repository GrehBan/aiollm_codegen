"""Contract snapshot folders become a buffered and a streamed method."""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from aiollm_codegen import contract_document, generate_package
from aiollm_codegen.cli import main


def _event(name: str) -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["type"],
        "properties": {"type": {"const": name}},
    }


def _snapshots(folder: Path, **changes: Any) -> Path:
    note = {"type": "object", "properties": {"text": {"type": "string"}}}
    parts: dict[str, Any] = {
        "request": {
            "title": "NotesCreateWireBody",
            "type": "object",
            "required": ["model"],
            "properties": {
                "model": {"type": "string"},
                "stream": {"type": "boolean"},
                "note": {"$ref": "#/$defs/Note"},
            },
            "$defs": {"Note": note},
        },
        "response": {
            "title": "Message",
            "type": "object",
            "required": ["id", "note"],
            "properties": {
                "id": {"type": "string"},
                "note": {"$ref": "#/$defs/Note"},
            },
            "$defs": {"Note": note},
        },
        "stream": {
            "anyOf": [
                {"$ref": "#/$defs/MessageStartEvent"},
                {"$ref": "#/$defs/MessageStopEvent"},
            ],
            "$defs": {
                "MessageStartEvent": _event("message_start"),
                "MessageStopEvent": _event("message_stop"),
                "Note": note,
            },
        },
    }
    parts.update(changes)
    folder.mkdir()
    for part, schema in parts.items():
        (folder / f"{part}.schema.json").write_text(
            json.dumps(schema), encoding="utf-8"
        )
    return folder


@pytest.fixture
def imported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[list[str]]:
    monkeypatch.syspath_prepend(str(tmp_path))
    names: list[str] = []
    yield names
    for module in [m for m in sys.modules if m.split(".")[0] in names]:
        sys.modules.pop(module)


def test_snapshots_generate_create_and_stream_methods(
    tmp_path: Path, imported: list[str]
) -> None:
    document = contract_document(_snapshots(tmp_path / "notes"), "/messages")
    files = generate_package(
        document, headers={"anthropic-version": "2026-10-21"}
    )
    assert "types/events.py" in files
    assert "methods/params/__init__.py" in files
    package = tmp_path / "notes_api"
    for path, code in files.items():
        (package / path).parent.mkdir(parents=True, exist_ok=True)
        (package / path).write_text(code, encoding="utf-8")
    imported.append("notes_api")
    api = importlib.import_module("notes_api")
    create = api.CreateMessage(model="m", note={"text": "hi"})
    stream = api.StreamMessage(model="m")
    assert create.request_path == stream.request_path == "/messages"
    assert stream.stream_terminal.matches({"type": "message_stop"})
    headers = stream.request_headers.copy()
    assert headers["anthropic-version"] == "2026-10-21"
    types = importlib.import_module("notes_api.types")
    assert types.MessageStreamEvent is not None
    reply = types.Message.model_validate({"id": "1", "note": {"text": "x"}})
    assert reply.note.text == "x"


@pytest.mark.parametrize(
    "path, changes, message",
    [
        ("messages", {}, "plain URL path"),
        ("/messages", {"request": {"type": "object"}}, "needs a title"),
        (
            "/messages",
            {
                "response": {
                    "title": "Message",
                    "type": "object",
                    "$defs": {"Note": {"type": "string"}},
                }
            },
            "disagree on definition Note",
        ),
    ],
)
def test_invalid_snapshots_are_rejected(
    tmp_path: Path, path: str, changes: dict[str, Any], message: str
) -> None:
    folder = _snapshots(tmp_path / "notes", **changes)
    with pytest.raises(ValueError, match=message):
        contract_document(folder, path)


def test_cli_contract_mode(tmp_path: Path) -> None:
    folder = _snapshots(tmp_path / "notes")
    output = tmp_path / "output"
    arguments = [str(folder), "--package", "--contract", "/messages"]
    assert main([*arguments, "-o", str(output)]) == 0
    assert "class StreamMessage(" in (
        output / "methods/messages.py"
    ).read_text("utf-8")
    with pytest.raises(SystemExit):
        main([str(folder), "--kind", "request", "--contract", "/x", "-o", "o"])
