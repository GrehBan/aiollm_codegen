"""Public generator APIs retain precise source and package return types."""

from __future__ import annotations

from pathlib import Path
from typing import assert_type

from aiollm_codegen import generate_models, generate_package


def check_api(source: Path) -> None:
    assert_type(generate_models(source, kind="request"), str)
    assert_type(
        generate_models(source, kind="response", schemas=["Response"]), str
    )
    assert_type(
        generate_package(
            source,
            operation_ids=["createResponse"],
            schema_groups={"Message": "messages"},
        ),
        dict[str, str],
    )
