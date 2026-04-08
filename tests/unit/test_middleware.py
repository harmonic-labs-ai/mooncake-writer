"""Unit tests for vLLM Mooncake trace capture middleware."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import mooncake_writer.middleware as middleware_module
from mooncake_writer.middleware import (
    ModelTraceRuntime,
    TraceConfig,
    VLLMMooncakeTraceMiddleware,
)


class FakeRuntime:
    """Runtime stub for middleware tests."""

    def __init__(
        self,
        *,
        rendered_prompt: str = "rendered prompt",
        hash_ids: list[int] | None = None,
        encoded_lengths: dict[str, int] | None = None,
        render_error: Exception | None = None,
        hash_error: Exception | None = None,
    ) -> None:
        self.rendered_prompt = rendered_prompt
        self.hash_ids = hash_ids or [11, 12]
        self.encoded_lengths = encoded_lengths or {}
        self.render_error = render_error
        self.hash_error = hash_error
        self.render_calls: list[dict[str, Any]] = []
        self.hash_prompts: list[str] = []

    def render_prompt(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> str:
        assert isinstance(messages, list)
        if tools is not None:
            assert isinstance(tools, list)
        if self.render_error is not None:
            raise self.render_error
        self.render_calls.append({"messages": messages, "tools": tools})
        return self.rendered_prompt

    def hash_prompt(self, text: str) -> list[int]:
        if self.hash_error is not None:
            raise self.hash_error
        self.hash_prompts.append(text)
        return list(self.hash_ids)

    def count_tokens(self, text: str) -> int | None:
        return self.encoded_lengths.get(text, len(text))


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
    runtime = FakeRuntime(
        hash_ids=[101, 202, 303],
        encoded_lengths={"rendered prompt": 5},
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
        runtime=runtime,
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
    assert runtime.hash_prompts == ["rendered prompt"]
    assert record["timestamp"] == 2_000
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
    runtime = FakeRuntime(
        hash_ids=[1],
        encoded_lengths={"rendered prompt": 4},
    )

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
        runtime=runtime,
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
    runtime = FakeRuntime(
        hash_ids=[404, 505],
        encoded_lengths={"rendered prompt": 4},
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
        runtime=runtime,
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
    runtime = FakeRuntime(
        hash_ids=[11, 22],
        encoded_lengths={"rendered prompt": 6},
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
        runtime=runtime,
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
    runtime = FakeRuntime(
        hash_ids=[7, 8, 9],
        encoded_lengths={"rendered prompt": 5},
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
        runtime=runtime,
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


def test_missing_chat_template_writes_hash_error(tmp_path: Path) -> None:
    path = tmp_path / "no-template.jsonl"
    request_payload = {
        "model": "demo-model",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    }
    runtime = FakeRuntime(render_error=RuntimeError("chat template unavailable"))

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        await receive()
        response = {
            "id": "chatcmpl-no-template",
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
        runtime=runtime,
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
    assert record["hash_ids"] == []
    assert record["hash_error"] == "RuntimeError: chat template unavailable"
    assert record["input_length"] == 4
    assert record["output_length"] == 2
    assert "hash_approximation" not in record


def test_hash_failures_write_hash_error(tmp_path: Path) -> None:
    path = tmp_path / "hash-error.jsonl"
    request_payload = {
        "model": "demo-model",
        "messages": [{"role": "user", "content": "hello"}],
    }
    runtime = FakeRuntime(hash_error=RuntimeError("boom"))

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
        runtime=runtime,
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
    slow_runtime = FakeRuntime(hash_ids=[1], encoded_lengths={"rendered prompt": 4})
    fast_runtime = FakeRuntime(hash_ids=[2], encoded_lengths={"rendered prompt": 4})

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
            runtime=slow_runtime,
            clock=lambda: 1_000,
        )
        fast = VLLMMooncakeTraceMiddleware(
            lambda scope, receive, send: fast_app(scope, receive, send, release_first),
            config=TraceConfig(enabled=True, path=path),
            runtime=fast_runtime,
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


def test_second_model_writes_hash_error_after_runtime_is_initialized(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    path = tmp_path / "multiple-models.jsonl"
    created: list[str] = []

    class FakeConstructedRuntime(FakeRuntime):
        def __init__(self, tokenizer_name: str, *, block_size: int) -> None:
            super().__init__(hash_ids=[31, 32], encoded_lengths={"rendered prompt": 4})
            created.append(tokenizer_name)

    monkeypatch.setattr(middleware_module, "ModelTraceRuntime", FakeConstructedRuntime)

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        await receive()
        response = {
            "id": "chatcmpl-runtime-check",
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
        clock=lambda: 10_000,
    )

    asyncio.run(
        _run_asgi_app(
            middleware,
            scope=_make_scope(),
            request_messages=_json_request_messages(
                {"model": "first-model", "messages": [{"role": "user", "content": "hello"}]}
            ),
        )
    )
    asyncio.run(
        _run_asgi_app(
            middleware,
            scope=_make_scope(),
            request_messages=_json_request_messages(
                {"model": "second-model", "messages": [{"role": "user", "content": "hello"}]}
            ),
        )
    )

    records = _read_jsonl(path)
    assert created == ["first-model"]
    assert records[0]["hash_ids"] == [31, 32]
    assert "hash_error" not in records[0]
    assert records[1]["hash_ids"] == []
    assert "multiple models/tokenizers are not supported" in records[1]["hash_error"]


def test_model_trace_runtime_uses_wrapped_chat_tokenizer(
    monkeypatch: Any,
) -> None:
    created: dict[str, Any] = {}
    chat_tokenizer = FakeChatTokenizer(rendered_prompt="wrapped prompt")

    class WrappedTokenizer:
        def __init__(self, underlying: FakeChatTokenizer) -> None:
            self._tokenizer = underlying
            self.encoded_texts: list[str] = []

        def encode(self, text: str) -> list[int]:
            self.encoded_texts.append(text)
            return list(range(len(text)))

    class FakeMooncakeWriter:
        def __init__(self, tokenizer_name: str, *, block_size: int) -> None:
            created["tokenizer_name"] = tokenizer_name
            created["block_size"] = block_size
            created["writer"] = self
            self.tokenizer = WrappedTokenizer(chat_tokenizer)
            self.hash_prompts: list[str] = []

        def text_to_hashes(self, text: str) -> list[int]:
            self.hash_prompts.append(text)
            return [77, 88]

    monkeypatch.setattr(middleware_module, "MooncakeWriter", FakeMooncakeWriter)

    runtime = ModelTraceRuntime("demo-model", block_size=64)
    prompt = runtime.render_prompt(
        [{"role": "user", "content": "hello"}],
        [{"type": "function", "function": {"name": "lookup"}}],
    )

    assert prompt == "wrapped prompt"
    assert runtime.hash_prompt(prompt) == [77, 88]
    assert runtime.count_tokens(prompt) == len("wrapped prompt")
    assert created["tokenizer_name"] == "demo-model"
    assert created["block_size"] == 64
    assert created["writer"].hash_prompts == ["wrapped prompt"]
