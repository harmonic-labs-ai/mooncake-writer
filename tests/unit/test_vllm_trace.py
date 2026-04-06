"""Unit tests for vLLM Mooncake trace capture middleware."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from mooncake_writer.vllm_trace import (
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
        error: Exception | None = None,
    ) -> None:
        self.hash_ids = hash_ids or [11, 12]
        self.tokenizer = FakeAIPerfTokenizer(encoded_lengths=encoded_lengths)
        self.error = error
        self.prompts: list[str] = []

    def text_to_hashes(self, text: str) -> list[int]:
        if self.error is not None:
            raise self.error
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


class NoChatTemplateTokenizer:
    """Tokenizer stub that forces the JSON fallback rendering path."""

    pass


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


def test_trace_uses_mooncake_writer(tmp_path: Path) -> None:
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
        config=TraceConfig(enabled=True, path=path),
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
    assert record["timestamp"] == 2_000
    assert record["input_length"] == 5
    assert record["output_length"] == 2
    assert record["hash_ids"] == [101, 202, 303]
    assert "captured_at_unix_ms" not in record
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
        config=TraceConfig(enabled=True, path=path),
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
    assert record["timestamp"] == 2_000
    assert record["request_id"] == "chatcmpl-response-id"


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
        config=TraceConfig(enabled=True, path=path),
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
    assert record["timestamp"] == 3_000
    assert record["request_id"] == "chatcmpl-stream"
    assert record["input_length"] == 4
    assert record["output_length"] == 2
    assert record["hash_ids"] == [404, 505]


def test_streaming_without_usage_omits_output_length(tmp_path: Path) -> None:
    path = tmp_path / "stream-no-usage.jsonl"
    request_payload = {
        "model": "demo-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    }
    fake_writer = FakeMooncakeWriter(
        hash_ids=[11, 22],
        encoded_lengths={"rendered prompt": 6},
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
            b"data: [DONE]\n\n",
        ]
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
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
        config=TraceConfig(enabled=True, path=path),
        writer_cache=writer_cache,
        clock=lambda: 4_000,
    )

    asyncio.run(
        _run_asgi_app(
            middleware,
            scope=_make_scope(),
            request_messages=_json_request_messages(request_payload),
        )
    )

    [record] = _read_jsonl(path)
    assert record["timestamp"] == 4_000
    assert record["request_id"] == "chatcmpl-stream"
    assert record["input_length"] == 6
    assert record["input_length_source"] == "rendered_prompt_tokenization"
    assert "output_length" not in record
    assert "output_length_source" not in record
    assert record["hash_ids"] == [11, 22]


def test_prompt_usage_missing_falls_back_to_rendered_prompt_tokenization(
    tmp_path: Path,
) -> None:
    path = tmp_path / "prompt-fallback.jsonl"
    request_payload = {
        "model": "demo-model",
        "messages": [{"role": "user", "content": "hello"}],
    }
    fake_writer = FakeMooncakeWriter(
        hash_ids=[7, 8, 9],
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
            "id": "chatcmpl-fallback",
            "usage": {"completion_tokens": 2},
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
        config=TraceConfig(enabled=True, path=path),
        writer_cache=writer_cache,
        clock=lambda: 5_000,
    )

    asyncio.run(
        _run_asgi_app(
            middleware,
            scope=_make_scope(),
            request_messages=_json_request_messages(request_payload),
        )
    )

    [record] = _read_jsonl(path)
    assert record["timestamp"] == 5_000
    assert record["request_id"] == "chatcmpl-fallback"
    assert record["input_length"] == 5
    assert record["input_length_source"] == "rendered_prompt_tokenization"
    assert record["output_length"] == 2
    assert record["output_length_source"] == "usage.completion_tokens"
    assert record["hash_ids"] == [7, 8, 9]


def test_invalid_json_request_is_skipped_but_replayed(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    seen: dict[str, bytes] = {}

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        chunks = []
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        seen["body"] = b"".join(chunks)
        await send(
            {
                "type": "http.response.start",
                "status": 400,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    middleware = VLLMMooncakeTraceMiddleware(
        app,
        config=TraceConfig(enabled=True, path=path),
        clock=lambda: 6_000,
    )

    asyncio.run(
        _run_asgi_app(
            middleware,
            scope=_make_scope(),
            request_messages=[{"type": "http.request", "body": b"{", "more_body": False}],
        )
    )

    assert seen["body"] == b"{"
    assert not path.exists()


def test_oversized_request_is_skipped_but_replayed(tmp_path: Path) -> None:
    path = tmp_path / "oversized.jsonl"
    request_body = json.dumps(
        {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
        }
    ).encode("utf-8")
    seen: dict[str, bytes] = {}

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        chunks = []
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        seen["body"] = b"".join(chunks)
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
                "body": b'{"id":"chatcmpl-skip"}',
                "more_body": False,
            }
        )

    middleware = VLLMMooncakeTraceMiddleware(
        app,
        config=TraceConfig(enabled=True, path=path, max_body_bytes=8),
        clock=lambda: 7_000,
    )

    asyncio.run(
        _run_asgi_app(
            middleware,
            scope=_make_scope(),
            request_messages=[{"type": "http.request", "body": request_body, "more_body": False}],
        )
    )

    assert seen["body"] == request_body
    assert not path.exists()


def test_chat_template_fallback_sets_hash_approximation(tmp_path: Path) -> None:
    path = tmp_path / "json-fallback.jsonl"
    request_payload = {
        "model": "demo-model",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    }
    fake_writer = FakeMooncakeWriter(hash_ids=[90, 91])
    writer_cache = FakeWriterCache(
        FakeWriterBundle(
            writer=fake_writer,
            chat_tokenizer=NoChatTemplateTokenizer(),
        )
    )

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        await receive()
        response = {
            "id": "chatcmpl-json-fallback",
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
        config=TraceConfig(enabled=True, path=path),
        writer_cache=writer_cache,
        clock=lambda: 8_000,
    )

    asyncio.run(
        _run_asgi_app(
            middleware,
            scope=_make_scope(),
            request_messages=_json_request_messages(request_payload),
        )
    )

    [record] = _read_jsonl(path)
    assert record["timestamp"] == 8_000
    assert record["hash_ids"] == [90, 91]
    assert "JSON fallback" in record["hash_approximation"]


def test_hash_failures_write_hash_error(tmp_path: Path) -> None:
    path = tmp_path / "hash-error.jsonl"
    request_payload = {
        "model": "demo-model",
        "messages": [{"role": "user", "content": "hello"}],
    }
    fake_writer = FakeMooncakeWriter(error=RuntimeError("boom"))
    writer_cache = FakeWriterCache(
        FakeWriterBundle(
            writer=fake_writer,
            chat_tokenizer=FakeChatTokenizer(rendered_prompt="rendered prompt"),
        )
    )

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        await receive()
        response = {
            "id": "chatcmpl-hash-error",
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
        config=TraceConfig(enabled=True, path=path),
        writer_cache=writer_cache,
        clock=lambda: 9_000,
    )

    asyncio.run(
        _run_asgi_app(
            middleware,
            scope=_make_scope(),
            request_messages=_json_request_messages(request_payload),
        )
    )

    [record] = _read_jsonl(path)
    assert record["timestamp"] == 9_000
    assert record["request_id"] == "chatcmpl-hash-error"
    assert record["hash_ids"] == []
    assert record["hash_error"] == "RuntimeError: boom"
    assert record["input_length"] == 4
    assert record["output_length"] == 2


def test_completion_order_can_differ_from_arrival_order(tmp_path: Path) -> None:
    path = tmp_path / "completion-order.jsonl"

    async def slow_app(
        scope: dict[str, Any],
        receive: Any,
        send: Any,
        release_first: asyncio.Event,
    ) -> None:
        await receive()
        await release_first.wait()
        response = {
            "id": "chatcmpl-slow",
            "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            "choices": [{"message": {"role": "assistant", "content": "a"}}],
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

    async def fast_app(
        scope: dict[str, Any],
        receive: Any,
        send: Any,
        release_first: asyncio.Event,
    ) -> None:
        await receive()
        response = {
            "id": "chatcmpl-fast",
            "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            "choices": [{"message": {"role": "assistant", "content": "b"}}],
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
        release_first.set()

    async def run_both() -> None:
        release_first = asyncio.Event()
        slow = VLLMMooncakeTraceMiddleware(
            lambda scope, receive, send: slow_app(scope, receive, send, release_first),
            config=TraceConfig(enabled=True, path=path),
            clock=lambda: 1_000,
        )
        fast = VLLMMooncakeTraceMiddleware(
            lambda scope, receive, send: fast_app(scope, receive, send, release_first),
            config=TraceConfig(enabled=True, path=path),
            clock=lambda: 2_000,
        )
        await asyncio.gather(
            _run_asgi_app(
                slow,
                scope=_make_scope(headers={"x-request-id": "req-slow"}),
                request_messages=_json_request_messages(
                    {"model": "demo-model", "messages": [{"role": "user", "content": "slow"}]}
                ),
            ),
            _run_asgi_app(
                fast,
                scope=_make_scope(headers={"x-request-id": "req-fast"}),
                request_messages=_json_request_messages(
                    {"model": "demo-model", "messages": [{"role": "user", "content": "fast"}]}
                ),
            ),
        )

    asyncio.run(run_both())

    records = _read_jsonl(path)
    assert [record["request_id"] for record in records] == ["req-fast", "req-slow"]
    assert [record["timestamp"] for record in records] == [2_000, 1_000]
