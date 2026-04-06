"""Unit tests for vLLM Mooncake trace capture middleware."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from mooncake_writer.vllm_trace import (
    OrderedJSONLTraceWriter,
    TraceConfig,
    VLLMMooncakeTraceMiddleware,
)


class FakeAIPerfTokenizer:
    """Tokenizer stub used by the fake MooncakeWriter."""

    def __init__(self, encoded_lengths: dict[str, int] | None = None) -> None:
        self.encoded_lengths = encoded_lengths or {}

    def encode(self, text: str) -> list[int]:
        return list(range(self.encoded_lengths.get(text, len(text))))


class FakeMooncakeWriter:
    """Small MooncakeWriter stub for middleware tests."""

    def __init__(
        self,
        *,
        hash_ids: list[int] | None = None,
        encoded_lengths: dict[str, int] | None = None,
    ) -> None:
        self.hash_ids = hash_ids or [11, 12]
        self.tokenizer = FakeAIPerfTokenizer(encoded_lengths=encoded_lengths)
        self.prompts: list[str] = []

    def text_to_hashes(self, text: str) -> list[int]:
        self.prompts.append(text)
        return list(self.hash_ids)


class FakeChatTokenizer:
    """Chat-template stub for rendering messages into prompt text."""

    def __init__(self, rendered_prompt: str = "rendered prompt") -> None:
        self.rendered_prompt = rendered_prompt

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        assert isinstance(messages, list)
        assert tokenize is False
        assert add_generation_prompt is True
        if tools is not None:
            assert isinstance(tools, list)
        return self.rendered_prompt


class FakeWriterBundle:
    """Writer bundle test double."""

    def __init__(self, writer: FakeMooncakeWriter, chat_tokenizer: FakeChatTokenizer) -> None:
        self.writer = writer
        self.chat_tokenizer = chat_tokenizer


class FakeWriterCache:
    """MooncakeWriter cache stub."""

    def __init__(self, bundle: FakeWriterBundle) -> None:
        self.bundle = bundle
        self.models: list[str] = []

    def get(self, model_name: str) -> FakeWriterBundle:
        self.models.append(model_name)
        return self.bundle


def _make_scope(*, headers: dict[str, str] | None = None) -> dict[str, Any]:
    raw_headers = [(b"content-type", b"application/json")]
    for key, value in (headers or {}).items():
        raw_headers.append((key.lower().encode("latin-1"), value.encode("latin-1")))
    return {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": raw_headers,
    }


async def _run_asgi_app(
    app: Any,
    *,
    scope: dict[str, Any],
    request_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sent_messages: list[dict[str, Any]] = []
    pending = list(request_messages)

    async def receive() -> dict[str, Any]:
        if pending:
            return pending.pop(0)
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent_messages.append(message)

    await app(scope, receive, send)
    return sent_messages


def _json_request_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    return [{"type": "http.request", "body": body, "more_body": False}]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line]


def test_writer_serializes_jsonl_record(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    writer = OrderedJSONLTraceWriter(path)

    first = writer.reserve_sequence()
    writer.complete(first, 1_000, {"request_id": "req-1", "model": "demo"})

    records = _read_jsonl(path)
    assert records == [
        {
            "captured_at_unix_ms": 1_000,
            "timestamp": 0,
            "request_id": "req-1",
            "model": "demo",
        }
    ]


def test_raw_mode_writes_messages_and_tools(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    request_payload = {
        "model": "demo-model",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "stream": False,
    }
    app_state: dict[str, Any] = {}

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        body_chunks = []
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            body_chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break

        app_state["request_body"] = b"".join(body_chunks)
        response = {
            "id": "chatcmpl-raw",
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            "choices": [{"message": {"role": "assistant", "content": "done"}}],
        }
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": json.dumps(response).encode("utf-8"),
                "more_body": False,
            }
        )

    middleware = VLLMMooncakeTraceMiddleware(
        app,
        config=TraceConfig(enabled=True, path=path, mode="raw"),
        clock=lambda: 5_000,
    )

    asyncio.run(
        _run_asgi_app(
            middleware,
            scope=_make_scope(headers={"x-request-id": "request-123"}),
            request_messages=_json_request_messages(request_payload),
        )
    )

    assert app_state["request_body"] == json.dumps(request_payload).encode("utf-8")

    [record] = _read_jsonl(path)
    assert record["timestamp"] == 0
    assert record["captured_at_unix_ms"] == 5_000
    assert record["request_id"] == "request-123"
    assert record["model"] == "demo-model"
    assert record["messages"] == request_payload["messages"]
    assert record["tools"] == request_payload["tools"]
    assert record["output_length"] == 3


def test_raw_mode_omits_input_length_and_hash_ids(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    request_payload = {
        "model": "demo-model",
        "messages": [{"role": "user", "content": "hello"}],
    }

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        await receive()
        response = {
            "id": "chatcmpl-raw",
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            "choices": [{"message": {"role": "assistant", "content": "done"}}],
        }
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": json.dumps(response).encode("utf-8"),
                "more_body": False,
            }
        )

    middleware = VLLMMooncakeTraceMiddleware(
        app,
        config=TraceConfig(enabled=True, path=path, mode="raw"),
        clock=lambda: 1_000,
    )

    asyncio.run(
        _run_asgi_app(
            middleware,
            scope=_make_scope(),
            request_messages=_json_request_messages(request_payload),
        )
    )

    [record] = _read_jsonl(path)
    assert "input_length" not in record
    assert "hash_ids" not in record


def test_hash_only_mode_uses_mooncake_writer(tmp_path: Path) -> None:
    path = tmp_path / "hash.jsonl"
    request_payload = {
        "model": "demo-model",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    }
    fake_writer = FakeMooncakeWriter(
        hash_ids=[101, 202, 303],
        encoded_lengths={"rendered prompt": 5},
    )
    writer_cache = FakeWriterCache(
        FakeWriterBundle(
            writer=fake_writer,
            chat_tokenizer=FakeChatTokenizer(rendered_prompt="rendered prompt"),
        )
    )

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        await receive()
        response = {
            "id": "chatcmpl-hash",
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        }
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": json.dumps(response).encode("utf-8"),
                "more_body": False,
            }
        )

    middleware = VLLMMooncakeTraceMiddleware(
        app,
        config=TraceConfig(enabled=True, path=path, mode="hash_only"),
        writer_cache=writer_cache,
        clock=lambda: 2_000,
    )

    asyncio.run(
        _run_asgi_app(
            middleware,
            scope=_make_scope(),
            request_messages=_json_request_messages(request_payload),
        )
    )

    [record] = _read_jsonl(path)
    assert writer_cache.models == ["demo-model"]
    assert fake_writer.prompts == ["rendered prompt"]
    assert record["input_length"] == 5
    assert record["output_length"] == 2
    assert record["hash_ids"] == [101, 202, 303]
    assert "messages" not in record
    assert "tools" not in record


def test_request_id_falls_back_to_response_id(tmp_path: Path) -> None:
    path = tmp_path / "request-id.jsonl"
    request_payload = {
        "model": "demo-model",
        "messages": [{"role": "user", "content": "hello"}],
    }

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        await receive()
        response = {
            "id": "chatcmpl-response-id",
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        }
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": json.dumps(response).encode("utf-8"),
                "more_body": False,
            }
        )

    middleware = VLLMMooncakeTraceMiddleware(
        app,
        config=TraceConfig(enabled=True, path=path, mode="raw"),
        clock=lambda: 2_000,
    )

    asyncio.run(
        _run_asgi_app(
            middleware,
            scope=_make_scope(),
            request_messages=_json_request_messages(request_payload),
        )
    )

    [record] = _read_jsonl(path)
    assert record["request_id"] == "chatcmpl-response-id"


def test_timestamp_is_relative_to_first_request(tmp_path: Path) -> None:
    path = tmp_path / "timestamps.jsonl"
    writer = OrderedJSONLTraceWriter(path)

    first = writer.reserve_sequence()
    second = writer.reserve_sequence()
    writer.complete(second, 1_300, {"request_id": "req-2"})
    writer.complete(first, 1_000, {"request_id": "req-1"})

    first_record, second_record = _read_jsonl(path)
    assert first_record["timestamp"] == 0
    assert second_record["timestamp"] == 300
    assert [first_record["request_id"], second_record["request_id"]] == [
        "req-1",
        "req-2",
    ]


def test_streaming_usage_chunk_is_captured_without_changing_sse(tmp_path: Path) -> None:
    path = tmp_path / "stream.jsonl"
    request_payload = {
        "model": "demo-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    }
    fake_writer = FakeMooncakeWriter(
        hash_ids=[404, 505],
        encoded_lengths={"rendered prompt": 4},
    )
    writer_cache = FakeWriterCache(
        FakeWriterBundle(
            writer=fake_writer,
            chat_tokenizer=FakeChatTokenizer(rendered_prompt="rendered prompt"),
        )
    )

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        await receive()
        chunks = [
            b'data: {"id":"chatcmpl-stream","choices":[{"delta":{"content":"hel"}}]}\n\n',
            b'data: {"id":"chatcmpl-stream","choices":[{"delta":{"content":"lo"}}]}\n\n',
            b'data: {"id":"chatcmpl-stream","choices":[],"usage":{"prompt_tokens":4,',
            b'"completion_tokens":2}}\n\n',
            b"data: [DONE]\n\n",
        ]
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream; charset=utf-8")],
            }
        )
        for chunk in chunks:
            await send(
                {
                    "type": "http.response.body",
                    "body": chunk,
                    "more_body": True,
                }
            )
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    middleware = VLLMMooncakeTraceMiddleware(
        app,
        config=TraceConfig(enabled=True, path=path, mode="hash_only"),
        writer_cache=writer_cache,
        clock=lambda: 3_000,
    )

    sent_messages = asyncio.run(
        _run_asgi_app(
            middleware,
            scope=_make_scope(),
            request_messages=_json_request_messages(request_payload),
        )
    )

    sent_body = b"".join(
        message.get("body", b"")
        for message in sent_messages
        if message["type"] == "http.response.body"
    )
    expected_body = (
        b'data: {"id":"chatcmpl-stream","choices":[{"delta":{"content":"hel"}}]}\n\n'
        b'data: {"id":"chatcmpl-stream","choices":[{"delta":{"content":"lo"}}]}\n\n'
        b'data: {"id":"chatcmpl-stream","choices":[],"usage":{"prompt_tokens":4,'
        b'"completion_tokens":2}}\n\n'
        b"data: [DONE]\n\n"
    )
    assert sent_body == expected_body

    [record] = _read_jsonl(path)
    assert record["request_id"] == "chatcmpl-stream"
    assert record["input_length"] == 4
    assert record["output_length"] == 2
    assert record["hash_ids"] == [404, 505]
