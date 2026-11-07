# Repository Guidelines

## Project Structure

This uv project generates aiollm-compatible Pydantic models and endpoint
methods from bundled specifications. Python 3.13 and Pydantic 2.14 are the
minimum targets. Public APIs: `generate_models`, `generate_package`;
CLI: `aiollm-codegen`.

Source lives in `src/aiollm_codegen/`. `schema.py` validates/selects inputs;
`normalization.py` adapts JSON Schema and composition; `generator.py` owns
backend configuration and AST adaptation. `response_unions.py` adds aiollm
unknown-variant handling; `operations.py` plans endpoints; `layout.py` groups
types; `package.py` emits methods. `publication.py` owns staged publication
and rollback. Keep transformations separate from I/O and reuse the backend
rather than building another schema compiler.

`examples/` contains input fixtures; `tests/` checks generated behavior.
`generated/` contains regenerable outputs. Provider snapshots live in the
adjacent `../aiollm/tests/providers/contracts/` checkout.

## Development Commands

Keep both repositories adjacent; install with `uv sync --all-groups --locked`.
Run the complete gate:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv run pyright
uv run pytest --mypy -m mypy tests/typing
uv run pytest --cov --cov-branch
uv run rumdl check README.md AGENTS.md
uv lock --check
uv pip check
uv build
```

These check style, typing, behavior/coverage, documentation, dependency
consistency and distributable packages. CI targets Windows and Linux.

## Style and Generation Contracts

Use four spaces, double quotes, future annotations, Google-style docstrings
and 79-character lines. Use descriptive PascalCase types and snake_case
modules. Apply aiollm's strict rules to source and generated output.

Requests use `BaseAPIType`, `MISSING` and separate wire aliases. Responses
preserve extras, accept future string enum values and use tagged unknown
fallbacks. Preserve `__upstream_schema__` identities. Emit real-value Literals,
Python type aliases, `Field(description=...)` and class `Attributes:` docs.
Follow aiollm's layout: received types under `types/`, endpoints under
`methods/` and request types under `methods/params/`. Stream methods
and binary downloads use `BaseStreamMethod`; uploads use aiollm `InputFile`.
Preserve aiollm's transport ownership; do not infer authentication or retries.

## Tests and Changes

Name tests `test_*.py`; reproduce defects before fixing them. Verify actual
payloads, serialization and local transport behavior without live credentials
or sleeps. Branch coverage must reach 98%. Generated contracts must pass
aiollm's actual Ruff, mypy and Pyright configurations. Never weaken checks.

Use descriptive `Area: change` commit titles. Keep reviews focused and report
checks actually run. Update examples and documented boundaries with behavior
changes. Do not commit secrets, overwrite edited generated packages, execute
generated code during generation or modify adjacent aiollm without permission.
