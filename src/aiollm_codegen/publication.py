"""Publish owned generated packages with rollback on failure."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from tempfile import mkdtemp

from pydantic import TypeAdapter

_MANIFEST = ".aiollm-codegen.json"
_HASHES = TypeAdapter(dict[str, str])


def _digest(path: Path) -> str:
    """Hash one owned file for detecting user edits before replacement."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_owned(destination: Path) -> None:
    """Reject user files, edits, symlinks, and unrecognized output folders."""
    manifest = destination / _MANIFEST
    if not manifest.is_file():
        raise ValueError("Output exists and is not an owned generated package")
    hashes = _HASHES.validate_json(manifest.read_text("utf-8"))
    for path in destination.rglob("*"):
        if path.is_symlink():
            raise ValueError("Generated packages must not contain symlinks")
        if not path.is_file() or path == manifest:
            continue
        relative = path.relative_to(destination).as_posix()
        if "__pycache__" in path.parts and path.suffix == ".pyc":
            continue
        if relative not in hashes or _digest(path) != hashes[relative]:
            raise ValueError(
                f"Output contains a user file or edit: {relative}"
            )


def _prune_folders(destination: Path, removed: set[Path]) -> None:
    """Delete folders left with only bytecode caches after file removal."""
    folders = {
        parent
        for relative in removed
        for parent in relative.parents
        if parent != Path()
    }
    for relative in sorted(folders, key=lambda p: len(p.parts), reverse=True):
        folder = destination / relative
        if folder.is_dir() and not any(
            path.is_file() and "__pycache__" not in path.parts
            for path in folder.rglob("*")
        ):
            shutil.rmtree(folder)


def _replace_files(package: Path, destination: Path, backup: Path) -> None:
    """Update owned files when Windows prevents renaming the directory."""
    shutil.copytree(destination, backup)
    previous = {
        path.relative_to(backup)
        for path in backup.rglob("*")
        if path.is_file()
    }
    staged = {
        path.relative_to(package)
        for path in package.rglob("*")
        if path.is_file()
    }
    changed: list[Path] = []
    try:
        for relative in sorted(
            staged, key=lambda p: (p.name == _MANIFEST, str(p))
        ):
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(package / relative, target)
            changed.append(relative)
        for relative in sorted(previous - staged):
            (destination / relative).unlink()
            changed.append(relative)
        _prune_folders(destination, previous - staged)
    except OSError as error:
        restored = True
        for relative in reversed(changed):
            try:
                if relative in previous:
                    (destination / relative).parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    shutil.copy2(backup / relative, destination / relative)
                else:
                    (destination / relative).unlink()
            except OSError as rollback:
                restored = False
                error.add_note(f"Failed to restore {relative}: {rollback}")
        if restored:
            shutil.rmtree(backup)
        else:
            error.add_note(f"Recovery backup retained at {backup}")
        raise
    shutil.rmtree(backup)


def _stage_package(files: Mapping[str, str], package: Path) -> None:
    """Write validated relative paths and the ownership manifest."""
    package.mkdir()
    hashes: dict[str, str] = {}
    for name, code in sorted(files.items()):
        relative = PurePosixPath(name)
        if (
            "\\" in name
            or ":" in name
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix != ".py"
        ):
            raise ValueError(f"Unsafe generated file path: {name}")
        path = package.joinpath(*relative.parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8", newline="\n")
        hashes[name] = _digest(path)
    (package / _MANIFEST).write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def publish_package(files: Mapping[str, str], destination: Path) -> None:
    """Stage a package and swap owned directories, rolling back on failure.

    Existing output is replaced only if its files match the generated
    manifest. Locked directories use per-file replacement with rollback.
    Readers must wait for publication. Failed rollback retains its backup.
    """
    if destination.is_symlink():
        raise ValueError("Output directory must not be a symlink")
    destination = destination.resolve()
    if destination.exists():
        _check_owned(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(mkdtemp(prefix=".aiollm-codegen-", dir=destination.parent))
    package = staging / "package"
    backup = staging / "backup"
    try:
        _stage_package(files, package)
        if destination.exists():
            try:
                os.replace(destination, backup)
            except PermissionError:
                _replace_files(package, destination, backup)
                return
        try:
            os.replace(package, destination)
        except OSError as error:
            if backup.exists():
                try:
                    os.replace(backup, destination)
                except OSError as rollback:
                    error.add_note(
                        f"Rollback failed: {rollback}; "
                        f"backup retained at {backup}"
                    )
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if not backup.exists():
            shutil.rmtree(staging)
