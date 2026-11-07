# aiollm-codegen

Generate Python 3.13+ Pydantic v2 **provider data models and endpoint methods**
compatible with
[`aiollm`](https://github.com/GrehBan/aiollm) from local OpenAPI 3.0/3.1
JSON or YAML files. Version 0.1 is experimental; review generated code
before adding it to a provider package.

## Setup

Keep the two repositories side by side:

```text
code/
  aiollm/
  aiollm_codegen/
```

```bash
cd aiollm_codegen
uv sync --all-groups --locked
```

The development environment installs `../aiollm` as an editable dependency
so tests verify the actual library contracts. The generator itself does not
import aiollm. To use generated modules elsewhere, install aiollm and
Pydantic 2.14+ in the target environment. Some schema formats may require
additional Pydantic dependencies, such as `pydantic[email]` for email fields.

## Generate models

These commands use the included example specification:

```bash
uv run aiollm-codegen examples/provider.openapi.yaml --kind request \
  --schema CreateRequest --output generated/requests.py
uv run aiollm-codegen examples/provider.openapi.yaml --kind response \
  --schema GenerationResponse --output generated/responses.py
```

Repeat `--schema` to select multiple components. Referenced dependencies are
included automatically. Without `--schema`, all components are generated
using the selected profile. Select request and response roots separately
when both appear in the same document. Existing output is replaced only
after successful generation; the input cannot be used as the output path.

## Generate a provider package

Package mode separates executable aiollm endpoint methods from reusable data
types and generates both request and response profiles:

```bash
uv run aiollm-codegen examples/provider.openapi.yaml --package \
  --operation createResponse --output generated/provider
```

```text
provider/
  __init__.py
  methods/
    __init__.py
    responses.py           # CreateResponse(BaseAPIMethod[GenerationResponse])
    params/                # request types, beside the methods that send them
      messages.py
      responses.py         # operation parameter mixin
      shared.py            # request data without a recognized family
  types/                   # received types, as in aiollm's provider packages
    __init__.py
    responses.py
    usage.py
```

As in aiollm, a module name names an API family: `types/responses.py` holds
the Responses API, not "response types". Methods are grouped by their first
OpenAPI tag, or the URL resource when no tag exists. An operation's
parameters and buffered result join its group (`ListModelsResponse` goes to
`types/models.py`). Other schema names determine stable type families:
`events`, `usage`, `tools`, `messages`, `responses`, `files`, `images`, and
`audio`. A type without a family joins the one family that uses it, or
`shared`. Inline types follow their owning component. Override a schema
group with a repeatable `--group SearchTool=search` option.

Repeat `--operation` to select operation IDs. Omitting it selects every
operation; unsupported operations fail generation with an explanation.
`--package` is separate from the existing `--kind`/`--schema` single-file mode.
For your OpenAI specification, a selection might be:

```bash
uv run aiollm-codegen examples/openai.openapi.yaml --package \
  --operation createResponse --operation listModels \
  --operation createFile --operation downloadFile -o generated/openai
```

Generated methods include the HTTP path, verb, query fields, body encoding
and response contract. JSON plus SSE endpoints produce both `CreateResponse`
and `StreamResponse`. Stream methods use `BaseStreamMethod`, typed SSE/NDJSON
items and `stream=True`; buffered companions accept only `stream=False` or
omission. Download methods stream `bytes`; uploads use aiollm `InputFile`
and multipart encoding. Retries remain disabled.

```python
from generated.openai import CreateFile, DownloadFile, StreamResponse

async with session.stream(StreamResponse(model="example", input="Hi")) as events:
    async for event in events:
        process(event)

await session.send_request(CreateFile(purpose="assistants", file=b"file data"))
async with session.stream(DownloadFile(file_id="file-1")) as download:
    await download.save("result.bin")
```

OpenAI Responses and Anthropic Messages streams use aiollm's terminal/error
event conventions. Other protocols can declare operation-level arrays
`x-aiollm-terminal-events` and `x-aiollm-error-events`. A terminal declaration
enables interruption detection; streams without one follow normal EOF rules.
SSE without an item schema yields raw `ServerSentEvent`; NDJSON without a
schema yields JSON values.
Types never depend on methods. Each profile is compiled once as a single
module, so class names stay unique and contextual (`EasyInputMessageRole`
rather than `Role`), and then split into snake_case family modules by
references. Mutually recursive types share a module. Families are ranked so
that modules import only lower-ranked ones; a type that needs a
higher-ranked family moves into it, so modules never import in a cycle.

A field whose type alias accepts null repeats it (`Metadata | None`), so the
annotation shows what the wire allows. In responses, a field that is
required but nullable may be left out and reads as `None`; providers omit
such fields. A URL with a `maxLength` becomes a string that keeps the limit.

Anthropic publishes no OpenAPI document; its methods come from aiollm's
docs-lite snapshot. `--contract PATH` reads a folder with
`request.schema.json`, `response.schema.json` and `stream.schema.json` as
`POST PATH`, named after the response (`CreateMessage`, `StreamMessage`,
`MessageStreamEvent`):

```bash
uv run aiollm-codegen ../aiollm/tests/providers/contracts/anthropic/messages \
  --package --contract /messages \
  --header anthropic-version=2026-10-21 -o generated/anthropic
```

Specifications that leave out aiollm details take explicit corrections:
`--name operationId=CreateMessage` sets a method class (its stream becomes
`StreamMessage`), `--stream operationId=EventSchema` adds a
`text/event-stream` variant beside the JSON result, and `--header` sets a
constant header on every method. A leading `/v1` in a path is dropped
because aiollm's `BaseAPI` adds the version. Optional header parameters are
not request fields; required ones are rejected.

Existing package output is replaceable only when its ownership manifest
matches its files. User edits, extra files, and symlinks are rejected. Staging
and directory swaps provide rollback on replacement failure. If Windows
holds the directory open, publication falls back to replacing owned files
with a backup and rollback; finish generation before importing the package.
If recovery also fails, the backup is retained and its location is reported. Regeneration
removes stale generated files and caches. Keep input specifications outside
the output package. A generated package is owned as a whole.

The package API returns `dict[str, str]`, mapping relative Python file paths
to source: `generate_package(path, operation_ids=["createResponse"],
schema_groups={"SearchTool": "search"})`.

## Generated data conventions

| Convention | Request profile | Response profile |
| --- | --- | --- |
| Base class | `BaseAPIType` | `BaseAPITypeExtra` |
| Omitted optional fields | `T \| MISSING = MISSING` | Backend defaults; a required nullable field may be absent (`None`) |
| Explicit null | Accepted only where the request schema permits it | Backend response nullability |
| Schema defaults | Not sent automatically; replaced with omission | Preserved |
| Unknown fields | Rejected (`extra="forbid"`) unless the schema allows them | Preserved, including nested models |
| Single-value tag (`type`) | Filled in by default and sent | Required as received |
| Wire names | `validation_alias` and `serialization_alias` | Same |

Named model classes get `__upstream_schema__` from their component identity;
inline models use `()`. Single-value enums become `Literal[value]`.
Request enums become `Literal` unions (`type ReasoningSummary =
Literal["auto", "concise", "detailed"]`), so plain strings type-check;
response enums accept future values as `str`.
Union, scalar, and container schemas use Python type aliases, such as
`type InputItem = Message | ToolCall`, instead of `RootModel` wrappers.
Validate a standalone alias with `TypeAdapter(InputItem).validate_python(...)`;
fields referencing it contain the underlying value directly, without `.root`.
Constraints and discriminator metadata remain in `Annotated` aliases.
Field descriptions use Pydantic `Field(description=...)`; class documentation
includes an `Attributes:` section. Generated docstrings decode literal `\n`
line breaks, trim blank edges, and
preserve paragraph and code indentation. Model data strings are unchanged.
The backend handles `$ref`, composition, enums, constraints, and recursive
types; this project adapts its output for aiollm. Tests cover representative
cases, rather than claiming full equivalence for every OpenAPI feature.

Generated request classes can serve as parameter mixins for
`BaseAPIMethod[ResponseModel]` and `BaseStreamMethod[Event]`. Package mode
generates these classes and resolves forward aliases before subclassing.
Authentication, retries and shared generation adapters remain provider
decisions.

The Python API returns source without writing a destination:

```python
from pathlib import Path

from aiollm_codegen import generate_models

source = generate_models(
    Path("examples/provider.openapi.yaml"),
    kind="request",
    schemas=["CreateRequest"],
)
```

## Input boundaries

- Unquoted YAML dates and timestamps remain strings, including model names,
  examples, and defaults. Other JSON scalar types keep their usual types.
- Supply reviewed, bundled specifications. References must point under
  `#/components/schemas/`; remote URLs and external files are rejected.
- Single-file mode accepts OpenAPI components or JSON Schema roots with
  `$defs`/`definitions`, in JSON or YAML. Package mode also extracts
  inline endpoint object schemas and resolves local parameter/request-body
  references. Profiles do not split read-only/write-only fields automatically.
- Response string enums accept future provider values; fixed discriminator
  constants remain literals. Tagged response unions use aiollm `BaseUnknown`
  and `by_type` fallbacks, preserving new variants and their extra fields.
  Known variants still validate against their required fields. Request enums
  remain closed. Inline type names include their parent model's name.
- Package mode supports required JSON or multipart object bodies, simple scalar
  path parameters, form/exploded query parameters, buffered JSON objects,
  plain text and empty responses, SSE/NDJSON event streams and binary downloads.
  Optional whole bodies, header/cookie parameters, alternate success-status
  shapes, non-object buffered JSON and multiple streaming formats on one
  endpoint are rejected. Each
  operation needs a unique `operationId`; parameters must match its path.
- Binary upload fields use aiollm `InputFile`; aiollm owns reading, closing,
  retrying and multipart encoding; downloads use aiollm's stream lifecycle and
  `Stream.save()`. The generator adds no replacement file I/O implementation.
- Dynamic references, schema IDs, and boolean component schemas are
  unsupported. Input is limited to 32 MiB and 100,000 visited nodes per
  component. These limits are guardrails, not a sandbox for hostile input.
- Generation does not execute generated Python. Tests import only code
  produced from reviewed fixtures. Validate real provider payloads against
  generated models before adopting them; advanced JSON Schema semantics
  remain subject to the backend's support.
- Deterministic output depends on the specification and locked toolchain.
  Inspect the diff when updating the generator dependency or `uv.lock`.

## Development

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv run pyright
uv run pytest --cov --cov-branch
uv run rumdl check README.md AGENTS.md
uv lock --check
uv pip check
uv build
```

Coverage includes branches with a 98% floor. Tests exercise emitted models,
actual aiollm method serialization, null/omission semantics, aliases,
unknown response fields, recursive types, discriminated unions, reference
selection, and CLI failure behavior.

## Design

[datamodel-code-generator](https://github.com/datamodel-code-generator/datamodel-code-generator)
owns schema parsing and Python type generation. `schema.py` owns local
input checks, component selection, and request default removal;
`generator.py` owns backend configuration and AST adaptation;
`operations.py`, `layout.py`, and `package.py` build the package plan.
`publication.py` owns directory publication and rollback; `cli.py`
owns arguments and single-file publication. A custom OpenAPI compiler
would duplicate complex reference and composition handling, so this
initial version deliberately reuses the established backend.
