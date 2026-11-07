"""Generated packages expose executable methods and grouped data contracts."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from aiohttp import BodyPartReader, web
from aiollm_codegen import OperationOverrides, generate_package
from aiollm_codegen.cli import main
from aiollm_codegen.publication import publish_package
from pydantic import MISSING, ValidationError

from aiollm import (
    BaseAPI,
    BaseErrorResolver,
    BaseStreamMethod,
    BearerAuth,
    BodyFormat,
    ResponseFormat,
    ServerSentEvent,
)
from aiollm.http.exceptions.status import APIStatusError
from aiollm.http.exceptions.transport import APIStreamInterruptedError
from aiollm.session.aiohttp import AiohttpSession
from aiollm.types.files import BytesFile

_RESULT_REF = "#/components/schemas/GenerationResponse"


def test_multipart_method_uses_aiollm_file_transport(
    api: dict[str, Any],
    write_api: Callable[[dict[str, Any]], Path],
    import_package: Callable[[dict[str, str]], Any],
) -> None:
    api["components"]["requestBodies"]["Create"] = {
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
    }
    package = import_package(
        generate_package(write_api(api), operation_ids=["createResponse"])
    )
    method = package.CreateResponse(item_id="upload", file=b"file data")
    assert method.request_body_format is BodyFormat.MULTIPART
    assert isinstance(method.file, BytesFile)

    async def exchange() -> None:
        received = []

        async def handle(request: web.Request) -> web.Response:
            reader = await request.multipart()
            part = await reader.next()
            assert isinstance(part, BodyPartReader)
            received.append((part.name, bytes(await part.read())))
            return web.json_response({"id": "file-1", "output": "uploaded"})

        app = web.Application()
        app.router.add_post("/responses/upload", handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        try:
            async with AiohttpSession(
                api=BaseAPI(f"http://127.0.0.1:{port}", ""),
                auth=BearerAuth("fake"),
                error_resolver=BaseErrorResolver(),
            ) as session:
                result = await session.send_request(method)
            assert result.output == "uploaded"
            assert received == [("file", b"file data")]
        finally:
            await runner.cleanup()

    asyncio.run(exchange())


def test_response_fallbacks_resolve_across_family_modules(
    api: dict[str, Any],
    write_api: Callable[[dict[str, Any]], Path],
    import_package: Callable[[dict[str, str]], Any],
) -> None:
    schemas = api["components"]["schemas"]
    for name, tag in (("MessageBlock", "text"), ("ToolBlock", "tool")):
        schemas[name] = {
            "type": "object",
            "required": ["type", "value"],
            "properties": {
                "type": {"const": tag},
                "value": {"type": "string"},
            },
        }
    schemas["GenerationResponse"]["properties"]["content"] = {
        "oneOf": [
            {"$ref": "#/components/schemas/MessageBlock"},
            {"$ref": "#/components/schemas/ToolBlock"},
        ],
        "discriminator": {"propertyName": "type"},
    }
    package = import_package(
        generate_package(write_api(api), operation_ids=["getResponse"])
    )
    response_type = package.GetResponse(item_id="x").response_type
    result = response_type.model_validate(
        {"id": "r1", "output": "ok", "content": {"type": "future", "new": 1}}
    )
    assert result.content.model_extra == {"new": 1}
    with pytest.raises(ValidationError):
        response_type.model_validate(
            {"id": "r1", "output": "ok", "content": {"type": "text"}}
        )


def test_generated_stream_methods_parse_events_and_detect_truncation(
    api: dict[str, Any],
    write_api: Callable[[dict[str, Any]], Path],
    import_package: Callable[[dict[str, str]], Any],
) -> None:
    operation = api["paths"]["/responses/{item-id}"]["post"]
    operation["x-aiollm-terminal-events"] = ["done"]
    operation["x-aiollm-error-events"] = ["broken"]
    operation["responses"]["200"]["content"]["text/event-stream"] = {
        "schema": {"$ref": "#/components/schemas/ResponseStreamEvent"}
    }
    schemas = api["components"]["schemas"]
    schemas.update(
        {
            "DeltaEvent": {
                "type": "object",
                "required": ["type", "delta"],
                "properties": {
                    "type": {"const": "delta"},
                    "delta": {"type": "string"},
                },
            },
            "DoneEvent": {
                "type": "object",
                "required": ["type"],
                "properties": {"type": {"const": "done"}},
            },
            "ResponseStreamEvent": {
                "oneOf": [
                    {"$ref": "#/components/schemas/DeltaEvent"},
                    {"$ref": "#/components/schemas/DoneEvent"},
                ],
                "discriminator": {"propertyName": "type"},
            },
        }
    )
    package = import_package(
        generate_package(write_api(api), operation_ids=["createResponse"])
    )
    method = package.StreamResponse(
        item_id="complete", model="test", messages=[]
    )
    assert method.get_body_map() == {
        "model": "test",
        "messages": [],
        "stream": True,
    }
    with pytest.raises(ValidationError):
        package.StreamResponse(
            item_id="x", model="test", messages=[], stream=False
        )
    with pytest.raises(ValidationError):
        package.CreateResponse(
            item_id="x", model="test", messages=[], stream=True
        )

    async def exchange() -> None:
        async def handle(request: web.Request) -> web.Response:
            assert request.headers["Accept"] == "text/event-stream"
            assert (await request.json())["stream"] is True
            events = [
                {"type": "delta", "delta": "Hi"},
                {"type": "future", "data": 1},
            ]
            if request.match_info["item"] == "complete":
                events.append({"type": "done"})
            elif request.match_info["item"] == "error":
                events = [{"type": "broken", "error": {"message": "failure"}}]
            return web.Response(
                text="".join(
                    "data: " + json.dumps(event) + "\n\n" for event in events
                ),
                content_type="text/event-stream",
            )

        app = web.Application()
        app.router.add_post("/responses/{item}", handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        try:
            async with AiohttpSession(
                api=BaseAPI(f"http://127.0.0.1:{port}", ""),
                auth=BearerAuth("fake"),
                error_resolver=BaseErrorResolver(),
            ) as session:
                async with session.stream(method) as stream:
                    events = [event async for event in stream]
                assert events[0].delta == "Hi"
                assert events[1].model_extra == {"data": 1}
                assert events[-1].type == "done"
                with pytest.raises(APIStreamInterruptedError):
                    async with session.stream(
                        package.StreamResponse(
                            item_id="truncated", model="test", messages=[]
                        )
                    ) as stream:
                        _ = [event async for event in stream]
                with pytest.raises(APIStatusError):
                    async with session.stream(
                        package.StreamResponse(
                            item_id="error", model="test", messages=[]
                        )
                    ) as stream:
                        _ = [event async for event in stream]
        finally:
            await runner.cleanup()

    asyncio.run(exchange())


def test_generated_file_download_streams_to_disk(
    api: dict[str, Any],
    write_api: Callable[[dict[str, Any]], Path],
    import_package: Callable[[dict[str, str]], Any],
    tmp_path: Path,
) -> None:
    api["paths"]["/files/content"] = {
        "get": {
            "operationId": "downloadFile",
            "tags": ["Files"],
            "responses": {
                "200": {
                    "content": {
                        "application/octet-stream": {
                            "schema": {"type": "string", "format": "binary"}
                        }
                    }
                }
            },
        }
    }
    package = import_package(
        generate_package(write_api(api), operation_ids=["downloadFile"])
    )
    method = package.DownloadFile()

    async def exchange() -> None:
        async def handle(request: web.Request) -> web.Response:
            return web.Response(
                body=b"file bytes", content_type="application/octet-stream"
            )

        app = web.Application()
        app.router.add_get("/files/content", handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        try:
            async with (
                AiohttpSession(
                    api=BaseAPI(f"http://127.0.0.1:{port}", ""),
                    auth=BearerAuth("fake"),
                    error_resolver=BaseErrorResolver(),
                ) as session,
                session.stream(method) as stream,
            ):
                assert await stream.save(tmp_path / "download.bin") == 10
            assert (tmp_path / "download.bin").read_bytes() == b"file bytes"
        finally:
            await runner.cleanup()

    asyncio.run(exchange())


def test_method_mixins_resolve_aliases_defined_later(
    api: dict[str, Any],
    write_api: Callable[[dict[str, Any]], Path],
    import_package: Callable[[dict[str, str]], Any],
) -> None:
    schemas = api["components"]["schemas"]
    schemas["Metadata"] = {
        "anyOf": [
            {"type": "object", "additionalProperties": {"type": "string"}},
            {"type": "null"},
        ]
    }
    schemas["CreateRequest"]["properties"]["metadata"] = {
        "$ref": "#/components/schemas/Metadata"
    }
    files = generate_package(
        write_api(api),
        operation_ids=["createResponse"],
        schema_groups=dict.fromkeys(schemas, "shared"),
    )
    package = import_package(files)
    method = package.CreateResponse(
        item_id="x", model="test", messages=[], metadata={"key": "value"}
    )
    assert method.get_body_map()["metadata"] == {"key": "value"}


@pytest.mark.parametrize(
    "media, schema, expected",
    [
        ("text/event-stream", None, "raw_sse"),
        ("application/x-ndjson", None, "raw_json"),
        (
            "application/ndjson",
            {
                "type": "object",
                "required": ["text"],
                "properties": {"text": {"type": "string"}},
            },
            "typed_json",
        ),
        ("audio/wav", None, "bytes"),
        ("application/pdf", {"type": "string", "format": "binary"}, "bytes"),
    ],
)
def test_stream_only_formats_use_correct_aiollm_contract(
    api: dict[str, Any],
    write_api: Callable[[dict[str, Any]], Path],
    import_package: Callable[[dict[str, str]], Any],
    media: str,
    schema: Any,
    expected: str,
) -> None:
    declaration = {} if schema is None else {"schema": schema}
    api["paths"]["/health"]["get"]["responses"] = {
        "200": {"content": {media: declaration}}
    }
    package = import_package(
        generate_package(write_api(api), operation_ids=["getHealth"])
    )
    method = package.GetHealth()
    assert isinstance(method, BaseStreamMethod)
    assert method.__request_headers__ == {"Accept": media}
    if expected == "bytes":
        assert method.response_type is bytes
        assert method.response_format is ResponseFormat.BYTES
    elif expected == "raw_sse":
        assert method.response_type is ServerSentEvent
        assert method.response_adapter is None
    else:
        assert method.response_format is ResponseFormat.NDJSON
        adapter = method.response_adapter
        assert adapter is not None
        value = adapter.validate_python({"text": "Hi"})
        if expected == "typed_json":
            assert value.text == "Hi"
        else:
            assert value == {"text": "Hi"}


@pytest.mark.parametrize(
    "path, event_name, identity, terminal",
    [
        (
            "/responses",
            "ResponseStreamEvent",
            "createResponse",
            "response.completed",
        ),
        ("/messages", "MessageStreamEvent", "createMessage", "message_stop"),
        (
            "/v1/responses",
            "ResponseStreamEvent",
            "createResponse",
            "response.incomplete",
        ),
        (
            "/v1/messages",
            "MessageStreamEvent",
            "createMessage",
            "message_stop",
        ),
    ],
)
def test_provider_stream_protocols_match_aiollm(
    api: dict[str, Any],
    write_api: Callable[[dict[str, Any]], Path],
    import_package: Callable[[dict[str, str]], Any],
    path: str,
    event_name: str,
    identity: str,
    terminal: str,
) -> None:
    operation = api["paths"]["/responses/{item-id}"]["post"]
    operation["operationId"] = identity
    operation["responses"]["200"]["content"]["text/event-stream"] = {
        "schema": {"$ref": "#/components/schemas/" + event_name}
    }
    api["components"]["schemas"][event_name] = {
        "type": "object",
        "properties": {"type": {"type": "string"}},
        "required": ["type"],
    }
    api["paths"] = {path: {"post": operation}}
    package = import_package(generate_package(write_api(api)))
    method_type = (
        package.StreamResponse
        if path.endswith("/responses")
        else package.StreamMessage
    )
    method = method_type(model="test", messages=[])
    assert method.stream_terminal.matches({"type": terminal})
    assert not method.stream_terminal.matches({"type": "future"})
    assert method.stream_parser.error_events.matches({"type": "error"})
    assert method.stream_parser.error_events.matches(
        {"type": "response.failed"}
    ) is path.endswith("/responses")


@pytest.mark.parametrize(
    "key, value",
    [
        ("x-aiollm-terminal-events", "done"),
        ("x-aiollm-error-events", [1]),
        ("x-aiollm-terminal-events", [""]),
    ],
)
def test_invalid_stream_policy_fails_before_publication(
    api: dict[str, Any],
    write_api: Callable[[dict[str, Any]], Path],
    key: str,
    value: Any,
) -> None:
    operation = api["paths"]["/health"]["get"]
    operation["responses"] = {"200": {"content": {"text/event-stream": {}}}}
    operation[key] = value
    with pytest.raises(ValueError, match=key):
        generate_package(write_api(api), operation_ids=["getHealth"])


def test_ambiguous_stream_formats_are_rejected(
    api: dict[str, Any],
    write_api: Callable[[dict[str, Any]], Path],
) -> None:
    api["paths"]["/health"]["get"]["responses"] = {
        "200": {
            "content": {
                "text/event-stream": {},
                "application/x-ndjson": {},
            }
        }
    }
    with pytest.raises(ValueError, match="Multiple streaming response"):
        generate_package(write_api(api), operation_ids=["getHealth"])


@pytest.fixture
def api() -> Any:
    return {
        "openapi": "3.1.0",
        "info": {"title": "Test", "version": "1"},
        "paths": {
            "/responses/{item-id}": {
                "parameters": [{"$ref": "#/components/parameters/ItemId"}],
                "post": {
                    "operationId": "createResponse",
                    "tags": ["Responses"],
                    "summary": "Create a response.",
                    "parameters": [
                        {
                            "in": "query",
                            "name": "limit",
                            "schema": {"type": "integer", "minimum": 1},
                        }
                    ],
                    "requestBody": {
                        "$ref": "#/components/requestBodies/Create"
                    },
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": _RESULT_REF}
                                }
                            }
                        }
                    },
                },
                "get": {
                    "operationId": "getResponse",
                    "tags": ["Responses"],
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": _RESULT_REF}
                                }
                            }
                        }
                    },
                },
                "delete": {
                    "operationId": "deleteResponse",
                    "responses": {"204": {"description": "Deleted"}},
                },
            },
            "/health": {
                "get": {
                    "operationId": "getHealth",
                    "responses": {
                        "200": {
                            "content": {
                                "text/plain": {"schema": {"type": "string"}}
                            }
                        }
                    },
                },
            },
        },
        "components": {
            "parameters": {
                "ItemId": {
                    "in": "path",
                    "name": "item-id",
                    "required": True,
                    "schema": {"type": "string"},
                }
            },
            "requestBodies": {
                "Create": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/CreateRequest"
                            }
                        }
                    },
                }
            },
            "schemas": {
                "CreateRequest": {
                    "type": "object",
                    "required": ["model", "messages"],
                    "properties": {
                        "model": {"type": "string"},
                        "messages": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/Message"},
                        },
                        "tool": {"$ref": "#/components/schemas/SearchTool"},
                        "stream": {"type": "boolean", "default": False},
                    },
                },
                "Message": {
                    "type": "object",
                    "required": ["text"],
                    "properties": {"text": {"type": "string"}},
                },
                "SearchTool": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
                "GenerationResponse": {
                    "type": "object",
                    "required": ["id", "output"],
                    "properties": {
                        "id": {"type": "string"},
                        "output": {"type": "string"},
                    },
                },
            },
        },
    }


@pytest.fixture
def write_api(tmp_path: Path) -> Any:
    def write(document: Any) -> Any:
        path = tmp_path / "api.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    return write


@pytest.fixture
def import_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    loaded = []
    monkeypatch.syspath_prepend(str(tmp_path))

    def load(files: Any) -> Any:
        name = f"generated_test_{len(loaded)}"
        publish_package(files, tmp_path / name)
        loaded.append(name)
        return importlib.import_module(name)

    yield load
    for name in list(sys.modules):
        if any(name == root or name.startswith(root + ".") for root in loaded):
            sys.modules.pop(name)


def test_package_groups_and_executes_methods(
    api: dict[str, Any],
    write_api: Callable[[dict[str, Any]], Path],
    import_package: Callable[[dict[str, str]], Any],
) -> None:
    source = write_api(api)
    files = generate_package(source)
    assert files == generate_package(source)
    assert "methods/params/messages.py" in files
    assert "methods/params/tools.py" in files
    assert "types/responses.py" in files
    assert "methods/responses.py" in files
    assert "BaseAPIMethod" not in "".join(
        code for path, code in files.items() if path.startswith("types/")
    )
    package = import_package(files)
    method = package.CreateResponse(
        item_id="one/two", model="test", messages=[{"text": "Hi"}], limit=2
    )
    assert method.request_path == "/responses/one%2Ftwo"
    assert method.get_query_map() == {"limit": "2"}
    assert method.get_body_map() == {
        "model": "test",
        "messages": [{"text": "Hi"}],
    }
    assert method.stream is MISSING
    assert method.retryable is False
    with pytest.raises(ValidationError):
        package.CreateResponse(
            item_id="a", model="test", messages=[], stream=True
        )
    assert package.GetHealth().response_type is str
    assert package.DeleteResponse(item_id="a").response_type is type(None)
    assert (
        package.GetResponse(item_id="a").request_body_format is BodyFormat.NONE
    )
    message = importlib.import_module(
        package.__name__ + ".methods.params.messages"
    ).Message
    assert message.__upstream_schema__ == "Message"
    assert package.__all__ == sorted(package.__all__)

    async def exchange() -> None:
        observed = []

        async def handle(request: web.Request) -> web.Response:
            observed.append(
                (
                    request.method,
                    request.path,
                    dict(request.query),
                    await request.json(),
                )
            )
            return web.json_response(
                {"id": "r1", "output": "ok", "new-field": 1}
            )

        app = web.Application()
        app.router.add_post("/responses/{item}", handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        try:
            async with AiohttpSession(
                api=BaseAPI(f"http://127.0.0.1:{port}", ""),
                auth=BearerAuth("fake"),
                error_resolver=BaseErrorResolver(),
            ) as session:
                result = await session.send_request(method)
            assert result.output == "ok"
            assert result.model_extra == {"new-field": 1}
            assert observed == [
                (
                    "POST",
                    "/responses/one/two",
                    {"limit": "2"},
                    {"model": "test", "messages": [{"text": "Hi"}]},
                )
            ]
        finally:
            await runner.cleanup()

    asyncio.run(exchange())


def test_explicit_group_and_inline_results(
    api: dict[str, Any],
    write_api: Callable[[dict[str, Any]], Path],
    import_package: Callable[[dict[str, str]], Any],
) -> None:
    api["paths"]["/responses/{item-id}"]["post"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"] = {
        "type": "object",
        "properties": {"output": {"type": "string"}},
    }
    files = generate_package(
        write_api(api),
        operation_ids=["createResponse"],
        schema_groups={"SearchTool": "search"},
    )
    assert "methods/params/search.py" in files
    assert "types/responses.py" in files
    assert "methods/health.py" not in files
    assert (
        import_package(files)
        .CreateResponse(item_id="x", model="m", messages=[])
        .response_type.__name__
        == "CreateResponseResult"
    )


def test_recursive_cross_group_models_preserve_aliases(
    api: dict[str, Any],
    write_api: Callable[[dict[str, Any]], Path],
    import_package: Callable[[dict[str, str]], Any],
) -> None:
    schemas = api["components"]["schemas"]
    schemas["Message"]["properties"]["tool"] = {
        "$ref": "#/components/schemas/SearchTool"
    }
    schemas["SearchTool"]["properties"]["message"] = {
        "$ref": "#/components/schemas/Message"
    }
    schemas["InputItem"] = {
        "oneOf": [
            {"$ref": "#/components/schemas/Message"},
            {"$ref": "#/components/schemas/SearchTool"},
        ]
    }
    schemas["CreateRequest"]["properties"]["item"] = {
        "$ref": "#/components/schemas/InputItem"
    }
    files = generate_package(write_api(api), operation_ids=["createResponse"])
    recursive = files["methods/params/messages.py"]
    assert "class Message(" in recursive and "class SearchTool(" in recursive
    assert not any("_internal" in path for path in files)
    assert "RootModel" not in "".join(files.values())
    package = import_package(files)
    method = package.CreateResponse(
        item_id="x",
        model="m",
        messages=[{"text": "Hi", "tool": {"message": {"text": "nested"}}}],
    )
    assert method.messages[0].tool.message.text == "nested"


def test_empty_text_and_bytes_responses(
    api: dict[str, Any],
    write_api: Callable[[dict[str, Any]], Path],
    import_package: Callable[[dict[str, str]], Any],
) -> None:
    operation = api["paths"]["/health"]["get"]
    for content, expected in (
        ("application/octet-stream", bytes),
        ("text/plain", str),
        (None, type(None)),
    ):
        operation["responses"] = (
            {"200": {"content": {content: {}}}} if content else {"204": {}}
        )
        files = generate_package(write_api(api), operation_ids=["getHealth"])
        assert [p for p in files if p.startswith("types/")] == [
            "types/__init__.py"
        ]
        assert import_package(files).GetHealth().response_type is expected


def test_cli_package_mode(
    api: dict[str, Any],
    write_api: Callable[[dict[str, Any]], Path],
    tmp_path: Path,
) -> None:
    source = write_api(api)
    arguments = [
        str(source),
        "--package",
        "--operation",
        "createResponse",
        "--group",
        "SearchTool=search",
        "--name",
        "createResponse=MakeResponse",
        "--header",
        "api-version=1",
        "-o",
        str(tmp_path / "output"),
    ]
    assert main(arguments) == 0
    assert (tmp_path / "output/methods/params/search.py").exists()
    methods = (tmp_path / "output/methods/responses.py").read_text("utf-8")
    assert "class MakeResponse(" in methods
    assert "api-version" in methods
    assert main(arguments) == 0


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--package", "--kind", "request"],
        ["--kind", "request", "--operation", "x"],
        ["--kind", "request", "--group", "A=b"],
        ["--kind", "request", "--stream", "a=B"],
        ["--package", "--schema", "A"],
    ],
)
def test_cli_rejects_incompatible_modes(
    tmp_path: Path, extra: list[str]
) -> None:
    with pytest.raises(SystemExit) as error:
        main(["input.yaml", "-o", str(tmp_path / "output"), *extra])
    assert error.value.code == 2


@pytest.mark.parametrize("flag", ["--group", "--name", "--stream", "--header"])
@pytest.mark.parametrize(
    "groups", [["broken"], ["=a"], ["A="], ["A=a", "A=b"]]
)
def test_cli_invalid_group_overrides(
    api: dict[str, Any],
    write_api: Callable[[dict[str, Any]], Path],
    tmp_path: Path,
    groups: list[str],
    flag: str,
) -> None:
    source = write_api(api)
    arguments = [str(source), "--package", "-o", str(tmp_path / "output")]
    for group in groups:
        arguments.extend([flag, group])
    assert main(arguments) == 1
    assert not (tmp_path / "output").exists()


def test_cli_refuses_source_within_package(
    api: dict[str, Any],
    write_api: Callable[[dict[str, Any]], Path],
    tmp_path: Path,
) -> None:
    source = write_api(api)
    assert main([str(source), "--package", "-o", str(tmp_path)]) == 1
    assert json.loads(source.read_text("utf-8")) == api


@pytest.mark.parametrize(
    "scope, path, value, message",
    [
        ("operation", ("parameters",), {}, "must be an array"),
        (
            "operation",
            ("parameters",),
            [{"in": "cookie", "name": "x"}],
            "Only named",
        ),
        ("parameter", ("style",), "matrix", "serialization"),
        ("parameter", ("explode",), True, "serialization"),
        ("parameter", ("allowReserved",), True, "serialization"),
        ("parameter", ("schema",), {"type": "array"}, "parameter type"),
        ("parameter", ("schema",), {"type": "object"}, "parameter type"),
        ("parameter", ("required",), False, "must be required"),
        (
            "operation",
            ("parameters",),
            [{"in": "query", "name": "item-id", "schema": {"type": "string"}}],
            "path and query",
        ),
        (
            "operation",
            ("parameters",),
            [{"in": "query", "name": "model", "schema": {"type": "string"}}],
            "overlap",
        ),
        (
            "operation",
            ("requestBody",),
            {"content": {"application/xml": {}}},
            "Only JSON",
        ),
        (
            "operation",
            ("requestBody",),
            {"content": {"application/json": {}}},
            "Optional request bodies",
        ),
        (
            "operation",
            ("requestBody",),
            {"$ref": "https://example.test/request"},
            "Unsupported operation reference",
        ),
        (
            "components",
            ("requestBodies", "Create"),
            {"$ref": "#/components/requestBodies/Create"},
            "Unsupported operation reference",
        ),
        (
            "components",
            ("schemas", "CreateRequest"),
            {"type": "string"},
            "must be object schemas",
        ),
        (
            "components",
            ("schemas", "CreateRequest"),
            {"allOf": [{"$ref": "#/components/schemas/CreateRequest"}]},
            "Cyclic allOf",
        ),
        ("operation", ("responses",), {"400": {}}, "No successful"),
        (
            "operation",
            ("responses",),
            {"200": {"content": {"text/csv": {}}}},
            "Unsupported response",
        ),
        (
            "operation",
            ("responses",),
            {"200": {}, "201": {"content": {"text/plain": {}}}},
            "Multiple success",
        ),
        ("parameter", ("name",), "other", "Path parameters differ"),
        ("operation", ("operationId",), "123", "Python class name"),
        ("operation", ("tags",), ["class"], "Invalid Python module"),
    ],
)
def test_unsupported_operations_fail_explicitly(
    api: dict[str, Any],
    write_api: Callable[[dict[str, Any]], Path],
    scope: str,
    path: Any,
    value: Any,
    message: str,
) -> None:
    target = {
        "operation": api["paths"]["/responses/{item-id}"]["post"],
        "parameter": api["components"]["parameters"]["ItemId"],
        "components": api["components"],
    }[scope]
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        generate_package(
            write_api(api),
            operation_ids=[
                api["paths"]["/responses/{item-id}"]["post"]["operationId"]
            ],
        )


@pytest.mark.parametrize(
    "identity", [None, "createResponse", "create-response"]
)
def test_missing_duplicate_and_normalized_operation_ids(
    api: dict[str, Any],
    write_api: Callable[[dict[str, Any]], Path],
    identity: Any,
) -> None:
    api["paths"]["/responses/{item-id}"]["get"]["operationId"] = identity
    with pytest.raises(
        ValueError, match="unique operationId|Operation names collide"
    ):
        generate_package(write_api(api))


def test_unknown_operation_and_group_names(
    api: dict[str, Any], write_api: Callable[[dict[str, Any]], Path]
) -> None:
    source = write_api(api)
    with pytest.raises(ValueError, match="No matching operations"):
        generate_package(source, operation_ids=["missing"])
    with pytest.raises(ValueError, match="Unknown schema group"):
        generate_package(source, schema_groups={"Missing": "group"})


def test_missing_ref_and_nonobject_operation_are_rejected(
    api: dict[str, Any], write_api: Callable[[dict[str, Any]], Path]
) -> None:
    operation = api["paths"]["/responses/{item-id}"]
    operation["post"]["requestBody"] = {
        "$ref": "#/components/requestBodies/Missing"
    }
    with pytest.raises(ValueError, match="Expected an object"):
        generate_package(write_api(api), operation_ids=["createResponse"])
    operation["post"] = None
    with pytest.raises(ValueError, match="Expected an object"):
        generate_package(write_api(api))


def test_inline_result_name_collision(
    api: dict[str, Any], write_api: Callable[[dict[str, Any]], Path]
) -> None:
    api["components"]["schemas"]["CreateResponseResult"] = {"type": "object"}
    api["paths"]["/responses/{item-id}"]["post"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"] = {"type": "object"}
    with pytest.raises(ValueError, match="Synthetic component name"):
        generate_package(write_api(api), operation_ids=["createResponse"])


@pytest.mark.parametrize("composition", [False, True])
def test_composition_parameter_overrides_and_alias_results(
    api: dict[str, Any],
    write_api: Callable[[dict[str, Any]], Path],
    import_package: Callable[[dict[str, str]], Any],
    composition: bool,
) -> None:
    schemas = api["components"]["schemas"]
    original = schemas["CreateRequest"]
    schemas["CreateRequest"] = {
        "allOf": [
            original,
            {"type": "object", "properties": {"extra": {"type": "string"}}},
        ]
    }
    schemas["ResponseAlias"] = (
        {"allOf": [{"$ref": _RESULT_REF}]}
        if composition
        else {"$ref": _RESULT_REF}
    )
    operation = api["paths"]["/responses/{item-id}"]["post"]
    operation["parameters"].append(
        {
            "in": "path",
            "name": "item-id",
            "required": True,
            "schema": {"type": "integer"},
        }
    )
    operation["responses"]["200"]["content"]["application/json"]["schema"] = {
        "$ref": "#/components/schemas/ResponseAlias"
    }
    package = import_package(
        generate_package(write_api(api), operation_ids=["createResponse"])
    )
    method = package.CreateResponse(item_id=42, model="m", messages=[])
    assert method.request_path == "/responses/42"
    result = method.response_type.model_validate({"id": "r1", "output": "ok"})
    assert result.id == "r1" and result.output == "ok"
    assert method.get_body_map() == {"model": "m", "messages": []}


def test_colliding_component_names_get_distinct_classes(
    api: dict[str, Any],
    write_api: Callable[[dict[str, Any]], Path],
    import_package: Callable[[dict[str, str]], Any],
) -> None:
    schemas = api["components"]["schemas"]
    schemas["Search-Tool"] = {
        "type": "object",
        "properties": {"depth": {"type": "integer"}},
    }
    schemas["CreateRequest"]["properties"]["secondTool"] = {
        "$ref": "#/components/schemas/Search-Tool"
    }
    files = generate_package(write_api(api), operation_ids=["createResponse"])
    package = import_package(files)
    method = package.CreateResponse(
        item_id="x", model="m", messages=[], second_tool={"depth": 2}
    )
    assert method.second_tool.depth == 2
    assert method.second_tool.__upstream_schema__ == "Search-Tool"
    exports = package.methods.params
    classes = {
        cls.__upstream_schema__: cls.__name__
        for cls in vars(exports).values()
        if isinstance(cls, type) and hasattr(cls, "__upstream_schema__")
    }
    assert {"SearchTool", "Search-Tool"} <= set(classes)
    assert classes["SearchTool"] != classes["Search-Tool"]


def test_discriminator_mapping_and_inline_types(
    api: dict[str, Any],
    write_api: Callable[[dict[str, Any]], Path],
    import_package: Callable[[dict[str, str]], Any],
) -> None:
    schemas = api["components"]["schemas"]
    schemas["Message"]["properties"]["type"] = {"const": "message"}
    schemas["Message"]["required"].append("type")
    schemas["SearchTool"]["properties"]["type"] = {"const": "tool"}
    schemas["SearchTool"]["required"] = ["type"]
    schemas["Item"] = {
        "oneOf": [
            {"$ref": "#/components/schemas/Message"},
            {"$ref": "#/components/schemas/SearchTool"},
        ],
        "discriminator": {
            "propertyName": "type",
            "mapping": {
                "message": "#/components/schemas/Message",
                "tool": "SearchTool",
            },
        },
    }
    schemas["CreateRequest"]["properties"]["item"] = {
        "$ref": "#/components/schemas/Item"
    }
    schemas["CreateRequest"]["properties"]["nested"] = {
        "type": "object",
        "properties": {"x": {"type": "string"}},
    }
    package = import_package(
        generate_package(write_api(api), operation_ids=["createResponse"])
    )
    method = package.CreateResponse(
        item_id="a",
        model="m",
        messages=[],
        item={"type": "tool"},
        nested={"x": "value"},
    )
    assert method.item.type == "tool"
    assert method.nested.__upstream_schema__ == ()
    with pytest.raises(ValidationError):
        package.CreateResponse(
            item_id="a", model="m", messages=[], item={"type": "unknown"}
        )


def test_overrides_name_methods_add_streams_and_constant_headers(
    api: dict[str, Any],
    write_api: Callable[[dict[str, Any]], Path],
    import_package: Callable[[dict[str, str]], Any],
) -> None:
    operation = api["paths"].pop("/responses/{item-id}")["post"]
    operation["parameters"] = [
        {"in": "header", "name": "api-version", "schema": {"type": "string"}}
    ]
    api["paths"] = {"/v1/messages": {"post": operation}}
    api["components"]["schemas"]["MessageStreamEvent"] = {
        "type": "object",
        "required": ["type"],
        "properties": {"type": {"type": "string"}},
    }
    api["components"]["schemas"]["CreateMessageParams"] = {"type": "object"}
    files = generate_package(
        write_api(api),
        overrides=OperationOverrides(
            names={"createResponse": "CreateMessage"},
            streams={"createResponse": "MessageStreamEvent"},
        ),
        headers={"api-version": "2023-06-01"},
    )
    package = import_package(files)
    create = package.CreateMessage(model="m", messages=[])
    stream = package.StreamMessage(model="m", messages=[])
    assert create.request_path == stream.request_path == "/messages"
    assert create.request_headers.copy()["api-version"] == "2023-06-01"
    assert stream.request_headers.copy()["Accept"] == "text/event-stream"
    assert stream.request_headers.copy()["api-version"] == "2023-06-01"
    assert stream.stream_terminal.matches({"type": "message_stop"})
    assert "api_version" not in type(create).model_fields


@pytest.mark.parametrize(
    "overrides, message",
    [
        (OperationOverrides(names={"missing": "X"}), "unselected operations"),
        (
            OperationOverrides(streams={"createResponse": "Missing"}),
            "Unknown stream event schema",
        ),
        (
            OperationOverrides(streams={"getHealth": "Message"}),
            "beside a JSON result",
        ),
    ],
)
def test_invalid_overrides_are_rejected(
    api: dict[str, Any],
    write_api: Callable[[dict[str, Any]], Path],
    overrides: OperationOverrides,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        generate_package(write_api(api), overrides=overrides)


def test_required_header_parameters_are_rejected(
    api: dict[str, Any], write_api: Callable[[dict[str, Any]], Path]
) -> None:
    api["paths"]["/health"]["get"]["parameters"] = [
        {
            "in": "header",
            "name": "x-key",
            "required": True,
            "schema": {"type": "string"},
        }
    ]
    with pytest.raises(ValueError, match="path/query parameters"):
        generate_package(write_api(api), operation_ids=["getHealth"])


def test_upstream_params_name_moves_method_parameters_aside(
    api: dict[str, Any],
    write_api: Callable[[dict[str, Any]], Path],
    import_package: Callable[[dict[str, str]], Any],
) -> None:
    schemas = api["components"]["schemas"]
    schemas["CreateResponseParams"] = {
        "type": "object",
        "properties": {"upstream": {"type": "string"}},
    }
    schemas["CreateRequest"]["properties"]["params"] = {
        "$ref": "#/components/schemas/CreateResponseParams"
    }
    files = generate_package(write_api(api), operation_ids=["createResponse"])
    method = import_package(files).CreateResponse(
        item_id="x", model="m", messages=[], params={"upstream": "u"}
    )
    assert method.params.upstream == "u"
    schemas["CreateResponseMethodParams"] = {"type": "object"}
    with pytest.raises(ValueError, match="Synthetic component name"):
        generate_package(write_api(api), operation_ids=["createResponse"])
