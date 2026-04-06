"""ASGI middleware for capturing vLLM chat traffic as Mooncake traces."""

from __future__ import annotations

import fcntl
import json
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mooncake_writer.writer import MooncakeWriter

ASGIMessage = dict[str, Any]
ASGIReceive = Callable[[], Any]
ASGISend = Callable[[ASGIMessage], Any]
ASGIApp = Callable[[dict[str, Any], ASGIReceive, ASGISend], Any]

DEFAULT_TRACE_PATH = "vllm_mooncake_traces.jsonl"
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
class BufferedRequestBody:
    """Buffered HTTP request body with replay support."""

    messages: list[ASGIMessage]
    body: bytes | None
    exceeded_limit: bool


class _JSONLTraceWriter:
    """Concurrency-safe JSONL appender."""

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


@dataclass(slots=True)
class WriterBundle:
    """Cached Mooncake writer plus the underlying chat tokenizer."""

    writer: MooncakeWriter
    chat_tokenizer: Any


class MooncakeWriterCache:
    """Lazy cache of MooncakeWriter instances keyed by model/tokenizer name."""

    def __init__(self, *, tokenizer_override: str | None = None, block_size: int) -> None:
        self._tokenizer_override = tokenizer_override
        self._block_size = block_size
        self._lock = threading.Lock()
        self._cache: dict[str, WriterBundle] = {}

    def get(self, model_name: str) -> WriterBundle:
        """Load or return a MooncakeWriter bundle for the requested model."""
        cache_key = self._tokenizer_override or model_name
        with self._lock:
            if cache_key not in self._cache:
                writer = MooncakeWriter(cache_key, block_size=self._block_size)
                chat_tokenizer = getattr(writer.tokenizer, "_tokenizer", writer.tokenizer)
                self._cache[cache_key] = WriterBundle(
                    writer=writer,
                    chat_tokenizer=chat_tokenizer,
                )
            return self._cache[cache_key]


class StreamSSEParser:
    """Incremental parser for ``text/event-stream`` response bodies."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> Iterator[str]:
        """Yield complete SSE ``data: ...`` payloads from a body chunk."""
        self._buffer.extend(chunk)
        while True:
            boundary = self._buffer.find(b"\n\n")
            if boundary < 0:
                break
            raw_event = bytes(self._buffer[:boundary])
            del self._buffer[: boundary + 2]
            data_lines = []
            for line in raw_event.splitlines():
                if line.startswith(b"data:"):
                    data_lines.append(line[5:].lstrip())
            if data_lines:
                yield b"\n".join(data_lines).decode("utf-8")


class ResponseCapture:
    """Collect response metadata without changing the client-visible payload."""

    def __init__(self) -> None:
        self.status_code: int | None = None
        self.content_type: str | None = None
        self.response_header_request_id: str | None = None
        self.response_id: str | None = None
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        self.output_length_source: str | None = None
        self.input_length_source: str | None = None
        self.non_streaming_body = bytearray()
        self._stream_parser = StreamSSEParser()

    @property
    def is_streaming(self) -> bool:
        """Return whether the downstream response is SSE."""
        return self.content_type == "text/event-stream"

    def observe(self, message: ASGIMessage) -> None:
        """Observe an ASGI response message."""
        if message["type"] == "http.response.start":
            self.status_code = message["status"]
            headers = _decode_headers(message.get("headers", []))
            self.content_type = headers.get("content-type", "").split(";", 1)[0].lower()
            self.response_header_request_id = headers.get("x-request-id")
            return

        if message["type"] != "http.response.body":
            return

        body = message.get("body", b"")
        if not body:
            return

        if self.is_streaming:
            for payload in self._stream_parser.feed(body):
                self._consume_stream_payload(payload)
        else:
            self.non_streaming_body.extend(body)

    def finalize(self) -> None:
        """Finish parsing any buffered non-streaming response state."""
        if self.is_streaming or not self.non_streaming_body:
            return

        try:
            payload = json.loads(self.non_streaming_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return

        self.response_id = _json_get_string(payload, "id")
        usage = payload.get("usage")
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            if isinstance(prompt_tokens, int):
                self.prompt_tokens = prompt_tokens
                self.input_length_source = "usage.prompt_tokens"
            if isinstance(completion_tokens, int):
                self.completion_tokens = completion_tokens
                self.output_length_source = "usage.completion_tokens"

    def _consume_stream_payload(self, payload: str) -> None:
        if payload == "[DONE]":
            return

        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            return

        if self.response_id is None:
            self.response_id = _json_get_string(chunk, "id")

        usage = chunk.get("usage")
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            if isinstance(prompt_tokens, int):
                self.prompt_tokens = prompt_tokens
                self.input_length_source = "usage.prompt_tokens"
            if isinstance(completion_tokens, int):
                self.completion_tokens = completion_tokens
                self.output_length_source = "usage.completion_tokens"


def _decode_headers(raw_headers: Iterable[tuple[bytes, bytes]]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in raw_headers:
        headers[key.decode("latin-1").lower()] = value.decode("latin-1")
    return headers


def _json_get_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) else None


def _count_tokens(bundle: WriterBundle, text: str) -> int | None:
    try:
        return len(bundle.writer.tokenizer.encode(text))
    except Exception:
        return None


def _render_prompt_text(
    bundle: WriterBundle,
    messages: list[Any],
    tools: list[Any] | None,
) -> tuple[str, str]:
    tokenizer = bundle.chat_tokenizer
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        if isinstance(rendered, str):
            return rendered, "chat_template"

    fallback = json.dumps(
        {"messages": messages, "tools": tools or []},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return fallback, "json_fallback"


async def _read_request_body(
    receive: ASGIReceive, *, max_body_bytes: int
) -> BufferedRequestBody:
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
        elif total_bytes > max_body_bytes:
            exceeded_limit = True
            body = bytearray()
            if message.get("more_body", False):
                break

        if not message.get("more_body", False):
            break

    return BufferedRequestBody(
        messages=messages,
        body=None if exceeded_limit else bytes(body),
        exceeded_limit=exceeded_limit,
    )


def _replay_receive(
    buffered_messages: Sequence[ASGIMessage], receive: ASGIReceive
) -> ASGIReceive:
    pending = list(buffered_messages)

    async def _inner() -> ASGIMessage:
        if pending:
            return pending.pop(0)
        return await receive()

    return _inner


def _parse_request_json(buffered_request: BufferedRequestBody) -> dict[str, Any] | None:
    if buffered_request.exceeded_limit or buffered_request.body is None:
        return None

    try:
        request_json = json.loads(buffered_request.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None

    return request_json if isinstance(request_json, dict) else None


def _build_trace_record(
    *,
    arrival_unix_ms: int,
    request_headers: dict[str, str],
    request_json: dict[str, Any],
    response_capture: ResponseCapture,
    writer_cache: MooncakeWriterCache,
) -> dict[str, Any] | None:
    if response_capture.status_code is None or response_capture.status_code >= 300:
        return None

    messages = request_json.get("messages")
    if not isinstance(messages, list):
        return None

    model_name = _json_get_string(request_json, "model")
    tools = request_json.get("tools") if isinstance(request_json.get("tools"), list) else None

    request_id = (
        request_headers.get("x-request-id")
        or response_capture.response_header_request_id
        or response_capture.response_id
        or uuid.uuid4().hex
    )

    record: dict[str, Any] = {"request_id": request_id, "timestamp": arrival_unix_ms}
    if model_name:
        record["model"] = model_name

    input_length = response_capture.prompt_tokens
    output_length = response_capture.completion_tokens
    prompt_token_count: int | None = None
    render_source: str | None = None
    hash_error: str | None = None

    if model_name:
        try:
            bundle = writer_cache.get(model_name)
            prompt_text, render_source = _render_prompt_text(bundle, messages, tools)
            record["hash_ids"] = bundle.writer.text_to_hashes(prompt_text)
            prompt_token_count = _count_tokens(bundle, prompt_text)

            if input_length is None and prompt_token_count is not None:
                input_length = prompt_token_count
                response_capture.input_length_source = "rendered_prompt_tokenization"
        except Exception as exc:
            record["hash_ids"] = []
            hash_error = f"{type(exc).__name__}: {exc}"
    else:
        record["hash_ids"] = []

    if input_length is not None:
        record["input_length"] = input_length
    if output_length is not None:
        record["output_length"] = output_length
    if response_capture.input_length_source:
        record["input_length_source"] = response_capture.input_length_source
    if response_capture.output_length_source:
        record["output_length_source"] = response_capture.output_length_source
    if render_source == "json_fallback":
        record["hash_approximation"] = (
            "hash_ids were generated from a JSON fallback because the tokenizer "
            "does not expose apply_chat_template(tokenize=False)"
        )
    elif (
        prompt_token_count is not None
        and input_length is not None
        and prompt_token_count != input_length
    ):
        record["hash_approximation"] = (
            "input_length uses vLLM usage.prompt_tokens while hash_ids use "
            "MooncakeWriter tokenization of the rendered prompt"
        )
    if hash_error is not None:
        record["hash_error"] = hash_error

    return record


class VLLMMooncakeTraceMiddleware:
    """ASGI middleware that captures vLLM chat completion traces."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        config: TraceConfig | None = None,
        writer_cache: MooncakeWriterCache | None = None,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self.app = app
        self.config = config or TraceConfig.from_env()
        self.writer = _JSONLTraceWriter(self.config.path)
        self.writer_cache = writer_cache or MooncakeWriterCache(
            tokenizer_override=self.config.tokenizer_name,
            block_size=self.config.block_size,
        )
        self.clock = clock or _wall_clock_ms

    async def __call__(
        self, scope: dict[str, Any], receive: ASGIReceive, send: ASGISend
    ) -> None:
        if not self._should_trace(scope):
            await self.app(scope, receive, send)
            return

        arrival_unix_ms = self.clock()
        request_headers = _decode_headers(scope.get("headers", []))
        buffered_request = await _read_request_body(
            receive, max_body_bytes=self.config.max_body_bytes
        )
        replay_receive = _replay_receive(buffered_request.messages, receive)
        request_json = _parse_request_json(buffered_request)
        if request_json is None:
            await self.app(scope, replay_receive, send)
            return

        response_capture = ResponseCapture()

        async def send_wrapper(message: ASGIMessage) -> None:
            response_capture.observe(message)
            await send(message)

        await self.app(scope, replay_receive, send_wrapper)
        response_capture.finalize()

        record = _build_trace_record(
            arrival_unix_ms=arrival_unix_ms,
            request_headers=request_headers,
            request_json=request_json,
            response_capture=response_capture,
            writer_cache=self.writer_cache,
        )
        self.writer.append(record)

    def _should_trace(self, scope: dict[str, Any]) -> bool:
        if not self.config.enabled:
            return False
        if scope.get("type") != "http":
            return False
        if scope.get("method") != "POST":
            return False
        return scope.get("path") == "/v1/chat/completions"


__all__ = [
    "MooncakeWriterCache",
    "TraceConfig",
    "VLLMMooncakeTraceMiddleware",
]
