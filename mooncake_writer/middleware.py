"""ASGI middleware for capturing vLLM chat traffic as Mooncake traces."""

from __future__ import annotations

import fcntl
import json
import os
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mooncake_writer.writer import MooncakeWriter

ASGIMessage = dict[str, Any]
ASGIReceive = Callable[[], Any]
ASGISend = Callable[[ASGIMessage], Any]
ASGIApp = Callable[[dict[str, Any], ASGIReceive, ASGISend], Any]

DEFAULT_TRACE_PATH = "/tmp/vllm_mooncake_trace.jsonl"
DEFAULT_BLOCK_SIZE = 512
DEFAULT_MAX_BODY_BYTES = 1024 * 1024


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _wall_clock_ms() -> int:
    return time.time_ns() // 1_000_000


def _parse_json_object(body: bytes) -> dict[str, Any] | None:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _decode_headers(raw_headers: Iterable[tuple[bytes, bytes]]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in raw_headers:
        headers[key.decode("latin-1").lower()] = value.decode("latin-1")
    return headers


def _json_get_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) else None


@dataclass(slots=True)
class TraceConfig:
    """Environment-driven configuration for trace capture."""

    enabled: bool = False
    path: Path = Path(DEFAULT_TRACE_PATH)
    block_size: int = DEFAULT_BLOCK_SIZE
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    tokenizer_name: str | None = None

    @classmethod
    def from_env(cls) -> "TraceConfig":
        """Load trace capture settings from environment variables."""
        return cls(
            enabled=_env_flag("VLLM_MOONCAKE_TRACE_ENABLED", False),
            path=Path(os.getenv("VLLM_MOONCAKE_TRACE_PATH", DEFAULT_TRACE_PATH)),
            block_size=_env_int(
                "VLLM_MOONCAKE_TRACE_BLOCK_SIZE", DEFAULT_BLOCK_SIZE
            ),
            max_body_bytes=_env_int(
                "VLLM_MOONCAKE_TRACE_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES
            ),
            tokenizer_name=os.getenv("VLLM_MOONCAKE_TRACE_TOKENIZER"),
        )


@dataclass(slots=True)
class CapturedRequest:
    """Buffered request body plus replayable ASGI messages."""

    messages: list[ASGIMessage]
    json_payload: dict[str, Any] | None

    @classmethod
    async def read(
        cls, receive: ASGIReceive, *, max_body_bytes: int
    ) -> "CapturedRequest":
        messages: list[ASGIMessage] = []
        body = bytearray()
        total_bytes = 0
        exceeded_limit = False

        while True:
            message = await receive()
            messages.append(message)
            if message["type"] != "http.request":
                break

            chunk = message.get("body", b"")
            total_bytes += len(chunk)
            if not exceeded_limit and total_bytes <= max_body_bytes:
                body.extend(chunk)
            else:
                exceeded_limit = True
                body.clear()
                if message.get("more_body", False):
                    break

            if not message.get("more_body", False):
                break

        payload = None if exceeded_limit else _parse_json_object(bytes(body))
        return cls(messages=messages, json_payload=payload)

    def replay(self, receive: ASGIReceive) -> ASGIReceive:
        pending = deque(self.messages)

        async def _replay() -> ASGIMessage:
            if pending:
                return pending.popleft()
            return await receive()

        return _replay


class TraceSink:
    """Concurrency-safe append-only JSONL sink."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        """Return the destination JSONL path."""
        return self._path

    def append(self, record: dict[str, Any] | None) -> None:
        """Append a single record when one was captured."""
        if record is None:
            return

        line = json.dumps(record, separators=(",", ":"), ensure_ascii=True)
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self._path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                os.write(fd, line.encode("utf-8") + b"\n")
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)


class SSEUsageExtractor:
    """Incrementally parse ``data: ...`` events from vLLM SSE responses."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> Iterator[dict[str, Any]]:
        self._buffer.extend(chunk)
        while True:
            raw_event = self._pop_event()
            if raw_event is None:
                break

            payload_text = self._extract_data(raw_event)
            if payload_text is None or payload_text == "[DONE]":
                continue

            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                continue

            if isinstance(payload, dict):
                yield payload

    def _pop_event(self) -> bytes | None:
        candidates = [
            (index, len(separator))
            for separator in (b"\n\n", b"\r\n\r\n")
            if (index := self._buffer.find(separator)) >= 0
        ]
        if not candidates:
            return None

        boundary, separator_size = min(candidates)
        raw_event = bytes(self._buffer[:boundary])
        del self._buffer[: boundary + separator_size]
        return raw_event

    def _extract_data(self, raw_event: bytes) -> str | None:
        data_lines = []
        for line in raw_event.splitlines():
            if not line or line.startswith(b":"):
                continue
            if line.startswith(b"data:"):
                data_lines.append(line[5:].lstrip())

        if not data_lines:
            return None

        try:
            return b"\n".join(data_lines).decode("utf-8")
        except UnicodeDecodeError:
            return None


class ObservedResponse:
    """Collect response metadata without changing the client-visible payload."""

    def __init__(self) -> None:
        self.status_code: int | None = None
        self.content_type: str | None = None
        self.response_header_request_id: str | None = None
        self.response_id: str | None = None
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        self.input_length_source: str | None = None
        self.output_length_source: str | None = None
        self._non_streaming_body = bytearray()
        self._sse_usage = SSEUsageExtractor()

    @property
    def is_streaming(self) -> bool:
        """Return whether the downstream response is SSE."""
        return self.content_type == "text/event-stream"

    def wrap(self, send: ASGISend) -> ASGISend:
        async def _wrapped(message: ASGIMessage) -> None:
            self.observe(message)
            await send(message)

        return _wrapped

    def observe(self, message: ASGIMessage) -> None:
        """Observe an ASGI response message."""
        if message["type"] == "http.response.start":
            self._observe_start(message)
            return
        if message["type"] == "http.response.body":
            self._observe_body(message)

    def finalize(self) -> None:
        """Finish parsing any buffered non-streaming response state."""
        if self.is_streaming or not self._non_streaming_body:
            return
        self._apply_payload(_parse_json_object(bytes(self._non_streaming_body)))

    def _observe_start(self, message: ASGIMessage) -> None:
        self.status_code = message["status"]
        headers = _decode_headers(message.get("headers", []))
        self.content_type = headers.get("content-type", "").split(";", 1)[0].lower()
        self.response_header_request_id = headers.get("x-request-id")

    def _observe_body(self, message: ASGIMessage) -> None:
        body = message.get("body", b"")
        if not body:
            return

        if self.is_streaming:
            for payload in self._sse_usage.feed(body):
                self._apply_payload(payload)
            return

        self._non_streaming_body.extend(body)

    def _apply_payload(self, payload: dict[str, Any] | None) -> None:
        if payload is None:
            return

        if self.response_id is None:
            self.response_id = _json_get_string(payload, "id")

        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return

        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if isinstance(prompt_tokens, int):
            self.prompt_tokens = prompt_tokens
            self.input_length_source = "usage.prompt_tokens"
        if isinstance(completion_tokens, int):
            self.completion_tokens = completion_tokens
            self.output_length_source = "usage.completion_tokens"


class ModelTraceRuntime:
    """Render prompts, hash them, and count tokens for a single model."""

    def __init__(self, tokenizer_name: str, *, block_size: int) -> None:
        self._writer = MooncakeWriter(tokenizer_name, block_size=block_size)
        self._chat_tokenizer = getattr(
            self._writer.tokenizer, "_tokenizer", self._writer.tokenizer
        )

    def render_prompt(self, messages: list[Any], tools: list[Any] | None) -> str:
        if not hasattr(self._chat_tokenizer, "apply_chat_template"):
            raise RuntimeError("chat template unavailable")

        try:
            rendered = self._chat_tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            rendered = self._chat_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        if not isinstance(rendered, str):
            raise RuntimeError("chat template did not return text")
        return rendered

    def hash_prompt(self, text: str) -> list[int]:
        return self._writer.text_to_hashes(text)

    def count_tokens(self, text: str) -> int | None:
        try:
            return len(self._writer.tokenizer.encode(text))
        except Exception:
            return None


@dataclass(slots=True)
class HashComputation:
    """Hash data derived from the rendered prompt."""

    hash_ids: list[int]
    input_length_fallback: int | None = None
    hash_error: str | None = None


def build_trace_record(
    *,
    arrival_ms: int,
    request_headers: dict[str, str],
    request_json: dict[str, Any],
    response: ObservedResponse,
    hash_result: HashComputation,
) -> dict[str, Any] | None:
    if response.status_code is None or response.status_code >= 300:
        return None

    messages = request_json.get("messages")
    if not isinstance(messages, list):
        return None

    request_id = (
        request_headers.get("x-request-id")
        or response.response_header_request_id
        or response.response_id
        or uuid.uuid4().hex
    )

    record: dict[str, Any] = {
        "timestamp": arrival_ms,
        "request_id": request_id,
        "hash_ids": hash_result.hash_ids,
    }
    if hash_result.hash_error is not None:
        record["hash_error"] = hash_result.hash_error

    if response.prompt_tokens is not None:
        record["input_length"] = response.prompt_tokens
        record["input_length_source"] = "usage.prompt_tokens"
    elif hash_result.input_length_fallback is not None:
        record["input_length"] = hash_result.input_length_fallback
        record["input_length_source"] = "rendered_prompt_tokenization"

    if response.completion_tokens is not None:
        record["output_length"] = response.completion_tokens
        record["output_length_source"] = "usage.completion_tokens"

    model_name = _json_get_string(request_json, "model")
    if model_name is not None:
        record["model"] = model_name

    return record


class VLLMMooncakeTraceMiddleware:
    """ASGI middleware that captures vLLM chat completion traces."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        config: TraceConfig | None = None,
        runtime: ModelTraceRuntime | None = None,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self.app = app
        self.config = config or TraceConfig.from_env()
        self.sink = TraceSink(self.config.path)
        self._runtime = runtime
        self._runtime_name = self.config.tokenizer_name
        self._runtime_lock = threading.Lock()
        self.clock = clock or _wall_clock_ms

    async def __call__(
        self, scope: dict[str, Any], receive: ASGIReceive, send: ASGISend
    ) -> None:
        if not self._should_trace(scope):
            await self.app(scope, receive, send)
            return

        arrival_ms = self.clock()
        request_headers = _decode_headers(scope.get("headers", []))
        captured = await CapturedRequest.read(
            receive,
            max_body_bytes=self.config.max_body_bytes,
        )
        replay_receive = captured.replay(receive)
        if captured.json_payload is None:
            await self.app(scope, replay_receive, send)
            return

        response = ObservedResponse()
        await self.app(scope, replay_receive, response.wrap(send))
        response.finalize()
        hash_result = self._compute_hash_result(
            captured.json_payload,
            existing_input_length=response.prompt_tokens,
        )

        record = build_trace_record(
            arrival_ms=arrival_ms,
            request_headers=request_headers,
            request_json=captured.json_payload,
            response=response,
            hash_result=hash_result,
        )
        self.sink.append(record)

    def _should_trace(self, scope: dict[str, Any]) -> bool:
        if not self.config.enabled:
            return False
        if scope.get("type") != "http":
            return False
        if scope.get("method") != "POST":
            return False
        return scope.get("path") == "/v1/chat/completions"

    def _compute_hash_result(
        self,
        request_json: dict[str, Any],
        *,
        existing_input_length: int | None,
    ) -> HashComputation:
        messages = request_json.get("messages")
        if not isinstance(messages, list):
            return HashComputation(hash_ids=[])

        tools = request_json.get("tools") if isinstance(request_json.get("tools"), list) else None

        try:
            runtime = self._get_runtime(request_json)
            if runtime is None:
                return HashComputation(hash_ids=[])

            prompt_text = runtime.render_prompt(messages, tools)
            hash_ids = runtime.hash_prompt(prompt_text)
            input_length_fallback = None
            if existing_input_length is None:
                input_length_fallback = runtime.count_tokens(prompt_text)
            return HashComputation(
                hash_ids=hash_ids,
                input_length_fallback=input_length_fallback,
            )
        except Exception as exc:
            return HashComputation(
                hash_ids=[],
                hash_error=f"{type(exc).__name__}: {exc}",
            )

    def _get_runtime(self, request_json: dict[str, Any]) -> ModelTraceRuntime | None:
        requested_runtime_name = self.config.tokenizer_name or _json_get_string(
            request_json, "model"
        )
        if requested_runtime_name is None:
            return None

        with self._runtime_lock:
            if self._runtime is None:
                self._runtime = ModelTraceRuntime(
                    requested_runtime_name,
                    block_size=self.config.block_size,
                )
                self._runtime_name = requested_runtime_name
                return self._runtime

            if self._runtime_name is None:
                self._runtime_name = requested_runtime_name
                return self._runtime

            if requested_runtime_name != self._runtime_name:
                raise RuntimeError(
                    "multiple models/tokenizers are not supported by this middleware "
                    "instance; set VLLM_MOONCAKE_TRACE_TOKENIZER to pin one tokenizer"
                )

            return self._runtime


__all__ = [
    "ModelTraceRuntime",
    "TraceConfig",
    "VLLMMooncakeTraceMiddleware",
]
