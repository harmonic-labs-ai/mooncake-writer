#!/usr/bin/env python3
"""Live end-to-end smoke tests for a vLLM server with Mooncake trace capture.

This script exercises the most important ways we interact with an OpenAI-
compatible vLLM server:

- `GET /v1/models`
- non-streaming `POST /v1/chat/completions`
- streaming `POST /v1/chat/completions`
- tool-bearing chat requests
- explicit request/session headers
- session IDs from the request body
- request-id fallback when no explicit `x-request-id` is provided
- invalid JSON error handling
- concurrent chat requests

When trace capture is enabled, it also validates the appended JSONL records.
Optionally, it can compare responses from a second baseline vLLM server.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TRACE_MODE_RAW = "raw"
TRACE_MODE_HASH_ONLY = "hash_only"


class TestFailure(RuntimeError):
    """Raised when a live-server assertion fails."""


@dataclass
class HTTPResult:
    """Container for an HTTP response."""

    status: int
    headers: dict[str, str]
    body: bytes


@dataclass
class StreamingResult:
    """Container for an SSE response."""

    status: int
    headers: dict[str, str]
    raw_body: str
    events: list[Any]


def _join_url(base_url: str, path: str) -> str:
    return urllib.parse.urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def _request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    timeout: float = 120.0,
) -> HTTPResult:
    request = urllib.request.Request(url, method=method, data=body)
    for key, value in (headers or {}).items():
        request.add_header(key, value)

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HTTPResult(
                status=response.status,
                headers={key.lower(): value for key, value in response.headers.items()},
                body=response.read(),
            )
    except urllib.error.HTTPError as exc:
        return HTTPResult(
            status=exc.code,
            headers={key.lower(): value for key, value in exc.headers.items()},
            body=exc.read(),
        )


def _request_json(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> HTTPResult:
    merged_headers = {"content-type": "application/json"}
    if headers:
        merged_headers.update(headers)
    return _request(
        "POST",
        _join_url(base_url, path),
        headers=merged_headers,
        body=json.dumps(payload).encode("utf-8"),
        timeout=timeout,
    )


def _request_sse(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> StreamingResult:
    merged_headers = {"content-type": "application/json"}
    if headers:
        merged_headers.update(headers)
    request = urllib.request.Request(
        _join_url(base_url, path),
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
    )
    for key, value in merged_headers.items():
        request.add_header(key, value)

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_parts: list[str] = []
            events: list[Any] = []
            data_lines: list[str] = []

            for raw_line in response:
                line = raw_line.decode("utf-8")
                raw_parts.append(line)

                stripped = line.rstrip("\r\n")
                if stripped == "":
                    if data_lines:
                        payload_text = "\n".join(data_lines)
                        if payload_text == "[DONE]":
                            events.append("[DONE]")
                        else:
                            try:
                                events.append(json.loads(payload_text))
                            except json.JSONDecodeError:
                                events.append(payload_text)
                        data_lines = []
                    continue

                if stripped.startswith("data:"):
                    data_lines.append(stripped[5:].lstrip())

            return StreamingResult(
                status=response.status,
                headers={key.lower(): value for key, value in response.headers.items()},
                raw_body="".join(raw_parts),
                events=events,
            )
    except urllib.error.HTTPError as exc:
        return StreamingResult(
            status=exc.code,
            headers={key.lower(): value for key, value in exc.headers.items()},
            raw_body=exc.read().decode("utf-8", errors="replace"),
            events=[],
        )


def _load_trace(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _wait_for_trace_count(path: Path, expected_count: int, timeout_seconds: float) -> list[dict[str, Any]]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        records = _load_trace(path)
        if len(records) >= expected_count:
            return records
        time.sleep(0.1)
    raise TestFailure(
        f"Timed out waiting for {expected_count} trace lines in {path}, "
        f"last observed count was {len(_load_trace(path))}"
    )


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise TestFailure(message)


def _content_text_from_non_stream(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return ""
    parts: list[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
    return "".join(parts)


def _content_text_from_stream(events: list[Any]) -> str:
    parts: list[str] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        choices = event.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if isinstance(content, str):
                parts.append(content)
    return "".join(parts)


def _normalize_non_stream(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": payload.get("model"),
        "usage": payload.get("usage"),
        "choices": payload.get("choices"),
    }


def _normalize_stream(events: list[Any]) -> dict[str, Any]:
    usage = None
    for event in events:
        if isinstance(event, dict) and isinstance(event.get("usage"), dict):
            usage = event["usage"]
    return {
        "content": _content_text_from_stream(events),
        "usage": usage,
        "event_count": len(events),
    }


def _set_body_path(payload: dict[str, Any], path: str, value: Any) -> None:
    current = payload
    parts = path.split(".")
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def _find_trace_by_request_id(records: list[dict[str, Any]], request_id: str) -> dict[str, Any]:
    for record in reversed(records):
        if record.get("request_id") == request_id:
            return record
    raise TestFailure(f"Could not find trace record for request_id={request_id!r}")


def _print_pass(label: str) -> None:
    print(f"[PASS] {label}")


def _validate_record_common(
    record: dict[str, Any],
    *,
    request_id: str,
    model: str,
    session_id: str | None = None,
) -> None:
    _assert(record.get("request_id") == request_id, f"Unexpected request_id in trace: {record}")
    _assert(record.get("model") == model, f"Unexpected model in trace: {record}")
    _assert(isinstance(record.get("timestamp"), int), f"Missing timestamp in trace: {record}")
    _assert(
        isinstance(record.get("captured_at_unix_ms"), int),
        f"Missing captured_at_unix_ms in trace: {record}",
    )
    if session_id is None:
        _assert("session_id" not in record, f"Unexpected session_id in trace: {record}")
    else:
        _assert(record.get("session_id") == session_id, f"Unexpected session_id in trace: {record}")


def _validate_raw_record(
    record: dict[str, Any],
    *,
    payload: dict[str, Any],
    request_id: str,
    model: str,
    session_id: str | None = None,
) -> None:
    _validate_record_common(record, request_id=request_id, model=model, session_id=session_id)
    _assert(record.get("messages") == payload["messages"], f"Raw trace missing messages: {record}")
    if "tools" in payload:
        _assert(record.get("tools") == payload["tools"], f"Raw trace missing tools: {record}")
    else:
        _assert("tools" not in record, f"Unexpected tools in raw trace: {record}")
    _assert("input_length" not in record, f"Raw trace should omit input_length: {record}")
    _assert("hash_ids" not in record, f"Raw trace should omit hash_ids: {record}")
    _assert(isinstance(record.get("output_length"), int), f"Raw trace missing output_length: {record}")


def _validate_hash_record(
    record: dict[str, Any],
    *,
    request_id: str,
    model: str,
    session_id: str | None = None,
) -> None:
    _validate_record_common(record, request_id=request_id, model=model, session_id=session_id)
    _assert("messages" not in record, f"Hash trace should omit messages: {record}")
    _assert("tools" not in record, f"Hash trace should omit tools: {record}")
    _assert(isinstance(record.get("input_length"), int), f"Hash trace missing input_length: {record}")
    _assert(isinstance(record.get("output_length"), int), f"Hash trace missing output_length: {record}")
    _assert(isinstance(record.get("hash_ids"), list), f"Hash trace missing hash_ids: {record}")


def _validate_mode_record(
    mode: str,
    record: dict[str, Any],
    *,
    payload: dict[str, Any],
    request_id: str,
    model: str,
    session_id: str | None = None,
) -> None:
    if mode == TRACE_MODE_RAW:
        _validate_raw_record(
            record,
            payload=payload,
            request_id=request_id,
            model=model,
            session_id=session_id,
        )
    else:
        _validate_hash_record(
            record,
            request_id=request_id,
            model=model,
            session_id=session_id,
        )


def _request_and_compare_non_stream(
    *,
    url: str,
    baseline_url: str | None,
    payload: dict[str, Any],
    headers: dict[str, str] | None,
    timeout: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    response = _request_json(url, "/v1/chat/completions", payload, headers=headers, timeout=timeout)
    _assert(response.status == 200, f"Non-stream request failed with {response.status}: {response.body!r}")
    response_json = json.loads(response.body.decode("utf-8"))

    baseline_json = None
    if baseline_url:
        baseline = _request_json(
            baseline_url,
            "/v1/chat/completions",
            payload,
            headers=headers,
            timeout=timeout,
        )
        _assert(
            baseline.status == response.status,
            f"Baseline status {baseline.status} did not match middleware status {response.status}",
        )
        baseline_json = json.loads(baseline.body.decode("utf-8"))
        _assert(
            _normalize_non_stream(baseline_json) == _normalize_non_stream(response_json),
            "Baseline and middleware non-stream responses differed after normalization",
        )

    return response_json, baseline_json


def _request_and_compare_stream(
    *,
    url: str,
    baseline_url: str | None,
    payload: dict[str, Any],
    headers: dict[str, str] | None,
    timeout: float,
) -> tuple[StreamingResult, StreamingResult | None]:
    response = _request_sse(url, "/v1/chat/completions", payload, headers=headers, timeout=timeout)
    _assert(response.status == 200, f"Streaming request failed with {response.status}: {response.raw_body!r}")

    baseline = None
    if baseline_url:
        baseline = _request_sse(
            baseline_url,
            "/v1/chat/completions",
            payload,
            headers=headers,
            timeout=timeout,
        )
        _assert(
            baseline.status == response.status,
            f"Baseline stream status {baseline.status} did not match middleware status {response.status}",
        )
        _assert(
            _normalize_stream(baseline.events) == _normalize_stream(response.events),
            "Baseline and middleware streaming responses differed after normalization",
        )

    return response, baseline


def _run_concurrent_requests(
    *,
    url: str,
    model: str,
    concurrency: int,
    timeout: float,
) -> list[tuple[str, dict[str, Any]]]:
    def _worker(index: int) -> tuple[str, dict[str, Any]]:
        request_id = f"req-concurrent-{index}"
        payload = {
            "model": model,
            "stream": False,
            "temperature": 0,
            "messages": [{"role": "user", "content": f"Concurrent ping {index}"}],
        }
        response = _request_json(
            url,
            "/v1/chat/completions",
            payload,
            headers={"x-request-id": request_id},
            timeout=timeout,
        )
        _assert(response.status == 200, f"Concurrent request failed with {response.status}")
        return request_id, payload

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_worker, index) for index in range(concurrency)]
        return [future.result() for future in futures]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True, help="Middleware server base URL, e.g. http://127.0.0.1:8000")
    parser.add_argument("--baseline-url", help="Optional baseline vLLM server URL without middleware")
    parser.add_argument("--model", required=True, help="Model name to send in requests")
    parser.add_argument("--trace-path", type=Path, help="Trace JSONL file written by the middleware server")
    parser.add_argument(
        "--trace-mode",
        choices=[TRACE_MODE_RAW, TRACE_MODE_HASH_ONLY],
        help="Expected trace mode for validation",
    )
    parser.add_argument(
        "--session-header",
        default="x-session-id",
        help="Session header name expected by the middleware",
    )
    parser.add_argument(
        "--session-body-path",
        help="Optional body path, e.g. metadata.session_id, to validate session capture from the request body",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="HTTP timeout for each request",
    )
    parser.add_argument(
        "--trace-wait-seconds",
        type=float,
        default=10.0,
        help="How long to wait for trace lines to flush",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent requests in the concurrency smoke test",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if (args.trace_path is None) != (args.trace_mode is None):
        raise TestFailure("--trace-path and --trace-mode must be provided together")

    trace_count = len(_load_trace(args.trace_path)) if args.trace_path else 0

    models_response = _request("GET", _join_url(args.server_url, "/v1/models"), timeout=args.timeout_seconds)
    _assert(models_response.status == 200, f"/v1/models failed with {models_response.status}")
    models_json = json.loads(models_response.body.decode("utf-8"))
    _assert(isinstance(models_json.get("data"), list), f"Unexpected /v1/models payload: {models_json}")
    if args.baseline_url:
        baseline_models = _request("GET", _join_url(args.baseline_url, "/v1/models"), timeout=args.timeout_seconds)
        _assert(
            baseline_models.status == models_response.status,
            f"Baseline /v1/models status {baseline_models.status} did not match {models_response.status}",
        )
    _print_pass("/v1/models")

    non_stream_request_id = "req-non-stream"
    non_stream_session_id = "sess-header-1"
    non_stream_payload = {
        "model": args.model,
        "stream": False,
        "temperature": 0,
        "messages": [{"role": "user", "content": "Say hello in five words."}],
    }
    non_stream_response, _ = _request_and_compare_non_stream(
        url=args.server_url,
        baseline_url=args.baseline_url,
        payload=non_stream_payload,
        headers={
            "x-request-id": non_stream_request_id,
            args.session_header: non_stream_session_id,
        },
        timeout=args.timeout_seconds,
    )
    _assert(
        isinstance(non_stream_response.get("usage"), dict),
        f"Non-stream response missing usage: {non_stream_response}",
    )
    _assert(
        _content_text_from_non_stream(non_stream_response) != "",
        f"Non-stream response missing assistant text: {non_stream_response}",
    )
    if args.trace_path:
        trace_count += 1
        records = _wait_for_trace_count(args.trace_path, trace_count, args.trace_wait_seconds)
        record = _find_trace_by_request_id(records, non_stream_request_id)
        _validate_mode_record(
            args.trace_mode,
            record,
            payload=non_stream_payload,
            request_id=non_stream_request_id,
            model=args.model,
            session_id=non_stream_session_id,
        )
    _print_pass("non-stream chat completion")

    stream_request_id = "req-stream"
    stream_payload = {
        "model": args.model,
        "stream": True,
        "temperature": 0,
        "messages": [{"role": "user", "content": "Explain Mooncake in one short sentence."}],
    }
    stream_response, _ = _request_and_compare_stream(
        url=args.server_url,
        baseline_url=args.baseline_url,
        payload=stream_payload,
        headers={"x-request-id": stream_request_id},
        timeout=args.timeout_seconds,
    )
    _assert(stream_response.events, "Streaming response emitted no SSE events")
    _assert(stream_response.events[-1] == "[DONE]", "Streaming response did not end with [DONE]")
    _assert(
        _content_text_from_stream(stream_response.events) != "",
        "Streaming response did not emit assistant text",
    )
    if args.trace_path:
        trace_count += 1
        records = _wait_for_trace_count(args.trace_path, trace_count, args.trace_wait_seconds)
        record = _find_trace_by_request_id(records, stream_request_id)
        _validate_mode_record(
            args.trace_mode,
            record,
            payload=stream_payload,
            request_id=stream_request_id,
            model=args.model,
        )
    _print_pass("streaming chat completion")

    tools_request_id = "req-tools"
    tools_payload = {
        "model": args.model,
        "stream": False,
        "temperature": 0,
        "messages": [{"role": "user", "content": "Decide whether to call a tool."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "description": "Look up the weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
    }
    tools_response, _ = _request_and_compare_non_stream(
        url=args.server_url,
        baseline_url=args.baseline_url,
        payload=tools_payload,
        headers={"x-request-id": tools_request_id},
        timeout=args.timeout_seconds,
    )
    _assert(isinstance(tools_response.get("usage"), dict), f"Tool response missing usage: {tools_response}")
    if args.trace_path:
        trace_count += 1
        records = _wait_for_trace_count(args.trace_path, trace_count, args.trace_wait_seconds)
        record = _find_trace_by_request_id(records, tools_request_id)
        _validate_mode_record(
            args.trace_mode,
            record,
            payload=tools_payload,
            request_id=tools_request_id,
            model=args.model,
        )
    _print_pass("tools request")

    if args.session_body_path:
        body_request_id = "req-session-body"
        body_session_id = "sess-body-1"
        body_payload = {
            "model": args.model,
            "stream": False,
            "temperature": 0,
            "messages": [{"role": "user", "content": "Capture the session id from the body."}],
        }
        _set_body_path(body_payload, args.session_body_path, body_session_id)
        body_response, _ = _request_and_compare_non_stream(
            url=args.server_url,
            baseline_url=args.baseline_url,
            payload=body_payload,
            headers={"x-request-id": body_request_id},
            timeout=args.timeout_seconds,
        )
        _assert(isinstance(body_response.get("usage"), dict), f"Session-body response missing usage: {body_response}")
        if args.trace_path:
            trace_count += 1
            records = _wait_for_trace_count(args.trace_path, trace_count, args.trace_wait_seconds)
            record = _find_trace_by_request_id(records, body_request_id)
            _validate_mode_record(
                args.trace_mode,
                record,
                payload=body_payload,
                request_id=body_request_id,
                model=args.model,
                session_id=body_session_id,
            )
        _print_pass("session id from body")

    fallback_payload = {
        "model": args.model,
        "stream": False,
        "temperature": 0,
        "messages": [{"role": "user", "content": "Request id should fall back to the response id."}],
    }
    fallback_response, _ = _request_and_compare_non_stream(
        url=args.server_url,
        baseline_url=args.baseline_url,
        payload=fallback_payload,
        headers=None,
        timeout=args.timeout_seconds,
    )
    fallback_request_id = fallback_response.get("id")
    _assert(isinstance(fallback_request_id, str), f"Response did not contain an id: {fallback_response}")
    if args.trace_path:
        trace_count += 1
        records = _wait_for_trace_count(args.trace_path, trace_count, args.trace_wait_seconds)
        record = _find_trace_by_request_id(records, fallback_request_id)
        _validate_mode_record(
            args.trace_mode,
            record,
            payload=fallback_payload,
            request_id=fallback_request_id,
            model=args.model,
        )
    _print_pass("request id fallback")

    invalid_body = _request(
        "POST",
        _join_url(args.server_url, "/v1/chat/completions"),
        headers={"content-type": "application/json"},
        body=b"{",
        timeout=args.timeout_seconds,
    )
    _assert(invalid_body.status >= 400, f"Invalid JSON unexpectedly succeeded: {invalid_body.status}")
    if args.trace_path:
        records = _wait_for_trace_count(args.trace_path, trace_count, args.trace_wait_seconds)
        _assert(
            len(records) == trace_count,
            "Invalid JSON request should not append a successful trace record",
        )
    _print_pass("invalid JSON error path")

    concurrent_requests = _run_concurrent_requests(
        url=args.server_url,
        model=args.model,
        concurrency=args.concurrency,
        timeout=args.timeout_seconds,
    )
    if args.trace_path:
        trace_count += args.concurrency
        records = _wait_for_trace_count(args.trace_path, trace_count, args.trace_wait_seconds)
        for request_id, payload in concurrent_requests:
            record = _find_trace_by_request_id(records, request_id)
            _validate_mode_record(
                args.trace_mode,
                record,
                payload=payload,
                request_id=request_id,
                model=args.model,
            )
        new_records = records[-args.concurrency :]
        timestamps = [record["timestamp"] for record in new_records]
        _assert(
            timestamps == sorted(timestamps),
            f"Concurrent trace timestamps were not monotonic: {timestamps}",
        )
    _print_pass("concurrent requests")

    print("All live vLLM checks passed.")
    if args.trace_path:
        print(f"Validated trace file: {args.trace_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TestFailure as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
