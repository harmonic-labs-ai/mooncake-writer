# Mooncake Writer Demo

A cloneable demo showing how to capture [Mooncake traces](https://github.com/kvcache-ai/Mooncake) using the `mooncake-writer` library (a small wrapper around [AIPerf](https://github.com/ai-dynamo/aiperf)).

The repo now includes two demo surfaces:

1. A notebook-oriented `MooncakeWriter` API for converting text to hash blocks
2. An importable ASGI middleware for capturing real `vLLM` OpenAI-compatible chat traffic

## Related Resources

- [Mooncake project](https://github.com/kvcache-ai/Mooncake)
- [Mooncake FAST '25 paper](https://www.usenix.org/conference/fast25/presentation/qin)
- [Mooncake technical report (arXiv)](https://arxiv.org/abs/2407.00079)
- [AIPerf repository](https://github.com/ai-dynamo/aiperf)

## What are Mooncake Traces?

A mooncake trace is a timestamped log of LLM requests designed for KV-cache simulation. Each record captures when a request arrived, how long the prompt was, how many tokens were generated, and a sequence of **hash block IDs** that represent the prompt content.

Hash blocks are fixed-size chunks of tokens mapped to unique integer IDs via a rolling hash. When two prompts share a common prefix, their hash block sequences share the same leading IDs — directly modelling KV-cache prefix overlap (cache hits). This lets simulators replay real traffic patterns and measure cache hit rates without needing the original prompt text.

The trace format is compatible with [AIPerf](https://github.com/ai-dynamo/aiperf)'s `MooncakeTrace` JSONL loader:

```json
{"timestamp": 1000, "input_length": 300, "output_length": 40, "hash_ids": [0, 1, 2]}
{"timestamp": 1500, "input_length": 150, "output_length": 20, "hash_ids": [0, 3]}
```

## Getting Started

### Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/)

### Setup

```bash
git clone https://github.com/harmonic-labs-ai/mooncake-writer
cd mooncake-writer
uv sync --all-extras
```

### Run the Demo

Open the notebook and step through the cells:

```bash
uv run jupyter notebook demo.ipynb
```

## Quick Example

```python
import time

from mooncake_writer import MooncakeWriter

writer = MooncakeWriter("gpt2", block_size=512)
now = int(time.time() * 1000)

writer.capture(
    "Write a short summary of Mooncake's KV-cache architecture.",
    timestamp_ms=now,
    output_length=64,
)
writer.capture(
    "Write a short summary of Mooncake's KV-cache architecture for AIPerf replay.",
    timestamp_ms=now + 100,
    output_length=64,
)

writer.write_trace("trace.jsonl")
```

## vLLM Chat Trace Capture

`mooncake_writer.middleware.VLLMMooncakeTraceMiddleware` captures `POST /v1/chat/completions`
traffic at the ASGI boundary and appends hash-only JSONL traces for later replay.
It renders the effective chat prompt, then hands that prompt to the existing
`MooncakeWriter` implementation in this repo.

### Enable It

Use the middleware with `vllm serve` and the request-ID / usage flags enabled:

```bash
export VLLM_MOONCAKE_TRACE_ENABLED=true

vllm serve Qwen/Qwen3-0.6B \
  --middleware mooncake_writer.middleware.VLLMMooncakeTraceMiddleware \
  --enable-request-id-headers \
  --enable-force-include-usage
```

### Config Knobs

| Variable | Default | Notes |
|---|---|---|
| `VLLM_MOONCAKE_TRACE_ENABLED` | `false` | Global on/off switch |
| `VLLM_MOONCAKE_TRACE_PATH` | `vllm_mooncake_traces.jsonl` | Append-only JSONL output |
| `VLLM_MOONCAKE_TRACE_BLOCK_SIZE` | `512` | Passed directly to `MooncakeWriter(block_size=...)` |
| `VLLM_MOONCAKE_TRACE_MAX_BODY_BYTES` | `1048576` | Requests above this size are passed through but not captured |
| `VLLM_MOONCAKE_TRACE_TOKENIZER` | unset | Optional tokenizer override for local prompt rendering |

### Sample Trace Line

```json
{"timestamp":1712400000000,"request_id":"req-123","model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","input_length":48,"output_length":17,"input_length_source":"usage.prompt_tokens","output_length_source":"usage.completion_tokens","hash_ids":[0,1,2]}
```

### Replay With AIPerf

Use the trace file when replaying with the Mooncake trace loader. These traces
store absolute arrival timestamps, so enable fixed-schedule auto-offset when replaying:

```bash
aiperf profile \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --endpoint-type chat \
  --streaming \
  --url http://127.0.0.1:8000/v1 \
  --input-file /tmp/vllm-chat-traces.jsonl \
  --custom-dataset-type mooncake_trace \
  --fixed-schedule \
  --fixed-schedule-auto-offset
```

### Live Server Test Script

Use the live smoke-test harness in [scripts/test_vllm_live_server.py](/Users/john/dev/mooncake-writer/scripts/test_vllm_live_server.py)
to exercise a real vLLM server end to end. It covers:

- `GET /v1/models`
- non-streaming chat completions
- streaming chat completions
- request-id fallback
- invalid JSON handling
- concurrent requests
- hash-only trace-file validation

Example against a middleware-enabled server:

```bash
python /Users/john/dev/mooncake-writer/scripts/test_vllm_live_server.py \
  --server-url http://127.0.0.1:8000 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --trace-path /tmp/vllm-chat-traces.jsonl
```

If you also have a baseline server without middleware, add `--baseline-url http://127.0.0.1:8001`
to compare normalized responses between the two servers.

### Limitations And Accuracy Notes

- Traces are always hash-only. They include derived token counts and `hash_ids`, not the original `messages` payload.
- `timestamp` is the Unix-millisecond arrival time captured when the request enters the middleware.
- The middleware uses exact `usage.prompt_tokens` / `usage.completion_tokens` when vLLM includes them, which is why `--enable-force-include-usage` is required for complete token counts.
- Streaming capture preserves the SSE stream verbatim and reads the final usage chunk when present. If usage is missing, `input_length` can fall back to local tokenization of the rendered prompt and `output_length` is omitted.
- The middleware assumes one effective tokenizer per server process. Set `VLLM_MOONCAKE_TRACE_TOKENIZER` when request `model` names are aliases or adapter names rather than directly loadable tokenizer IDs.
- The middleware uses this repo's `MooncakeWriter.text_to_hashes()` over the rendered chat prompt. That means the hash IDs follow this repo's existing MooncakeWriter semantics rather than vLLM's internal block hash implementation.
- If the local tokenizer cannot render a chat template, the middleware still writes the trace record but leaves `hash_ids` empty and records a `hash_error`.
- v1 writes a single append-only file. It does not include built-in rotation.
- Records are appended when requests complete, so on-disk line order may differ from arrival order. Replay should use `--fixed-schedule-auto-offset`.

## What's in the Notebook

The [`demo.ipynb`](demo.ipynb) notebook walks through:

1. **Initialising** a `MooncakeWriter` with a HuggingFace tokenizer
2. **Converting text to hash blocks** — tokenize, chunk, and hash a single string
3. **Reconstructing text from hashes** — round-trip back to text
4. **Batch operations** — convert multiple texts at once and observe shared prefix hash IDs
5. **Custom block sizes** — adjust the token-per-block granularity
6. **Cross-request prefix sharing** — detect shared prefixes across separate calls
7. **Trace capture** — record timestamped requests and write aiperf-compatible JSONL

## API Summary

| Method | Description |
|---|---|
| `MooncakeWriter(tokenizer, block_size=512)` | Create a writer with a tokenizer name or instance |
| `text_to_hashes(text)` | Single text to hash ID list |
| `texts_to_hashes(texts)` | Batch texts to hash ID lists |
| `hashes_to_text(hash_ids, input_length)` | Hash IDs back to a single text string |
| `hashes_to_texts(hash_ids_list, input_lengths)` | Batch hash IDs back to text strings |
| `capture(text, timestamp_ms, output_length=None)` | Hash text and record a timestamped trace entry |
| `write_trace(path)` | Write captured traces to a JSONL file |
| `clear_trace()` | Clear the trace buffer |
| `reset_hashes()` | Reset the internal hasher (clears hash-to-ID mappings) |

`text_to_hashes`, `texts_to_hashes`, `hashes_to_text`, and `hashes_to_texts` accept an optional `block_size` override.
