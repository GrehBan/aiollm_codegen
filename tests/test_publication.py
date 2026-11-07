"""Owned directory publication preserves user files and recovers failures."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pytest
from aiollm_codegen.publication import publish_package


def test_regeneration_removes_stale_generated_files_but_protects_edits(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "output"
    publish_package({"__init__.py": "", "old.py": "old = 1\n"}, destination)
    cache = destination / "__pycache__"
    cache.mkdir()
    (cache / "old.pyc").write_bytes(b"cached")
    publish_package({"__init__.py": "", "new.py": "new = 2\n"}, destination)
    assert not (destination / "old.py").exists()
    assert not cache.exists()
    (destination / "new.py").write_text("user edit", encoding="utf-8")
    with pytest.raises(ValueError, match="user file or edit"):
        publish_package({"__init__.py": ""}, destination)
    assert (destination / "new.py").read_text("utf-8") == "user edit"


def test_user_directories_and_extra_files_are_preserved(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "output"
    destination.mkdir()
    with pytest.raises(ValueError, match="not an owned"):
        publish_package({"__init__.py": ""}, destination)
    destination.rmdir()
    publish_package({"__init__.py": ""}, destination)
    (destination / "notes.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="user file or edit"):
        publish_package({"__init__.py": ""}, destination)
    assert (destination / "notes.txt").read_text("utf-8") == "keep"


@pytest.mark.parametrize(
    "name",
    ["../outside.py", "/absolute.py", "C:\\outside.py", "a\\b.py", "file.txt"],
)
def test_unsafe_output_paths_never_publish(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="Unsafe generated file"):
        publish_package({name: ""}, tmp_path / "output")
    assert list(tmp_path.iterdir()) == []


def test_failed_replacement_restores_previous_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "output"
    publish_package({"__init__.py": "old = 1\n"}, destination)
    replace = os.replace

    def fail_new_package(source: Any, target: Path) -> Any:
        if Path(source).name == "package":
            raise PermissionError("locked")
        replace(source, target)

    monkeypatch.setattr(os, "replace", fail_new_package)
    with pytest.raises(PermissionError, match="locked"):
        publish_package({"__init__.py": "new = 2\n"}, destination)
    assert (destination / "__init__.py").read_text("utf-8") == "old = 1\n"
    assert list(tmp_path.iterdir()) == [destination]


def test_failed_rollback_retains_backup_for_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "output"
    publish_package({"__init__.py": "old = 1\n"}, destination)
    replace = os.replace

    def fail_new_and_restore(source: Any, target: Path) -> Any:
        if Path(source).name in ("package", "backup"):
            raise PermissionError("locked")
        replace(source, target)

    monkeypatch.setattr(os, "replace", fail_new_and_restore)
    with pytest.raises(PermissionError) as error:
        publish_package({"__init__.py": "new = 2\n"}, destination)
    assert "backup retained at" in "\n".join(error.value.__notes__)
    backup = next(tmp_path.glob(".aiollm-codegen-*/backup/__init__.py"))
    assert backup.read_text("utf-8") == "old = 1\n"


def test_output_and_nested_symlinks_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "output"
    publish_package({"__init__.py": ""}, destination)
    original = Path.is_symlink
    for target, message in (
        (destination, "must not be a symlink"),
        (destination / "__init__.py", "must not contain symlinks"),
    ):

        def is_link(path: Path, target: Path = target) -> bool:
            return path == target or original(path)

        monkeypatch.setattr(Path, "is_symlink", is_link)
        with pytest.raises(ValueError, match=message):
            publish_package({"__init__.py": ""}, destination)


def test_regeneration_when_windows_holds_directory_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "output"
    publish_package(
        {"__init__.py": "old = 1\n", "stale.py": "stale = 1\n"}, destination
    )
    replace = os.replace

    def deny_directory_rename(source: Any, target: Path) -> Any:
        if Path(source) == destination:
            raise PermissionError("directory is open")
        replace(source, target)

    monkeypatch.setattr(os, "replace", deny_directory_rename)
    publish_package(
        {"__init__.py": "new = 2\n", "new.py": "new = 2\n"}, destination
    )
    assert (destination / "__init__.py").read_text("utf-8") == "new = 2\n"
    assert (destination / "new.py").exists()
    assert not (destination / "stale.py").exists()
    assert list(tmp_path.iterdir()) == [destination]


def test_locked_directory_fallback_removes_stale_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "output"
    publish_package(
        {"__init__.py": "", "events/__init__.py": "", "events/a.py": ""},
        destination,
    )
    cache = destination / "events" / "__pycache__"
    cache.mkdir()
    (cache / "a.cpython-313.pyc").write_bytes(b"cached")
    replace = os.replace

    def deny_directory_rename(source: Any, target: Path) -> Any:
        if Path(source) == destination:
            raise PermissionError("directory is open")
        replace(source, target)

    monkeypatch.setattr(os, "replace", deny_directory_rename)
    publish_package({"__init__.py": "", "events.py": ""}, destination)
    assert (destination / "events.py").exists()
    assert not (destination / "events").exists()


@pytest.mark.parametrize("rollback_failure", [False, True])
def test_locked_file_fallback_restores_output_or_retains_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rollback_failure: bool
) -> None:
    destination = tmp_path / "output"
    publish_package(
        {"__init__.py": "old = 1\n", "z.py": "old = 1\n"}, destination
    )
    replace = os.replace

    def deny_directory_and_last_file(source: Any, target: Path) -> Any:
        if Path(source) == destination or Path(source).name == "z.py":
            raise PermissionError("locked")
        replace(source, target)

    monkeypatch.setattr(os, "replace", deny_directory_and_last_file)
    if rollback_failure:

        def deny_restore(*args: Any, **kwargs: Any) -> Any:
            raise PermissionError("restore locked")

        monkeypatch.setattr(shutil, "copy2", deny_restore)
    with pytest.raises(PermissionError, match="locked") as error:
        publish_package(
            {
                "__init__.py": "new = 2\n",
                "new.py": "new = 2\n",
                "z.py": "new = 2\n",
            },
            destination,
        )
    assert (destination / "z.py").read_text("utf-8") == "old = 1\n"
    assert not (destination / "new.py").exists()
    if rollback_failure:
        assert "backup retained at" in "\n".join(error.value.__notes__)
        assert (
            next(
                tmp_path.glob(".aiollm-codegen-*/backup/__init__.py")
            ).read_text("utf-8")
            == "old = 1\n"
        )
    else:
        assert (destination / "__init__.py").read_text("utf-8") == "old = 1\n"
        assert list(tmp_path.iterdir()) == [destination]
