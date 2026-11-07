"""Format and lint emitted Python using aiollm's strict Ruff profile."""

from __future__ import annotations

import ast
import io
import subprocess
import sys
import tokenize
from importlib.resources import as_file, files
from itertools import accumulate
from typing import Final

from aiollm_codegen.docstrings import format_docstrings

_STRING_WIDTH: Final = 48
_LINE_LENGTH: Final = 79
_MIN_WIDTH: Final = 20


def _chunks(value: str, width: int = _STRING_WIDTH) -> list[str]:
    """Split long strings without modifying any runtime characters.

    Chunks end after whitespace where possible, so words stay whole; a
    single word longer than a chunk is split by characters.
    """
    result: list[str] = []
    current = ""
    for character in value:
        if current and len(repr(current + character)) > width:
            boundary = max(current.rfind(" "), current.rfind("\n")) + 1
            if 0 < boundary < len(current) and not character.isspace():
                result.append(current[:boundary])
                current = current[boundary:]
            else:
                result.append(current)
                current = ""
        current += character
    result.append(current)
    return result


def _wrap_strings(code: str) -> str:
    """Wrap long metadata and wire strings with implicit concatenation.

    A string is wrapped when it is long or its line is too long; chunks
    are sized for the indentation the formatter gives them.
    """
    documents = {
        (node.lineno, node.col_offset)
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }
    offsets = list(
        accumulate(map(len, code.splitlines(keepends=True)), initial=0)
    )
    replacements: list[tuple[int, int, str]] = []
    for token in tokenize.generate_tokens(io.StringIO(code).readline):
        if token.type != tokenize.STRING or token.start in documents:
            continue
        line = token.line.splitlines()[0] if token.line else ""
        if (
            len(token.string) <= _STRING_WIDTH
            and len(line.rstrip()) <= _LINE_LENGTH
        ):
            continue
        value = ast.literal_eval(token.string)
        if not isinstance(value, str):
            continue
        indent = len(line) - len(line.lstrip())
        width = max(_MIN_WIDTH, min(_STRING_WIDTH, _LINE_LENGTH - indent - 6))
        literal = (
            "(\n"
            + "\n".join(repr(chunk) for chunk in _chunks(value, width))
            + "\n)"
        )
        replacements.append(
            (
                offsets[token.start[0] - 1] + token.start[1],
                offsets[token.end[0] - 1] + token.end[1],
                literal,
            )
        )
    for start, end, literal in reversed(replacements):
        code = code[:start] + literal + code[end:]
    return code


def _ruff(code: str, command: list[str], config: str) -> str:
    """Run a pinned formatter with explicit configuration and no shell."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            *command,
            "--config",
            config,
            "--stdin-filename",
            "models.py",
            "-",
        ],
        input=code,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ValueError(
            "Generated code fails aiollm Ruff checks:\n" + result.stderr
        )
    return result.stdout


def format_source(code: str) -> str:
    """Format emitted Python using the strict upstream lint profile."""
    code = format_docstrings(code)
    with as_file(files("aiollm_codegen").joinpath("strict.toml")) as config:
        code = _wrap_strings(_ruff(code, ["format"], str(config)))
        code = _ruff(code, ["format"], str(config))
        code = _ruff(code, ["check", "--fix", "--unsafe-fixes"], str(config))
        return _ruff(code, ["format"], str(config))
