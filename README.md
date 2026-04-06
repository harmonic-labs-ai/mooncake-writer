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

`mooncake_writer.vllm_trace.VLLMMooncakeTraceMiddleware` captures `POST /v1/chat/completions`
traffic at the ASGI boundary and appends JSONL traces for later replay. In
`hash_only` mode it renders the effective chat prompt, then hands that prompt to
the existing `MooncakeWriter` implementation in this repo.

### Enable It

Use the middleware with `vllm serve` and the request-ID / usage flags enabled:

```bash
export VLLM_MOONCAKE_TRACE_ENABLED=true
export VLLM_MOONCAKE_TRACE_PATH=/tmp/vllm-chat-traces.jsonl
export VLLM_MOONCAKE_TRACE_MODE=hash_only
export VLLM_MOONCAKE_TRACE_BLOCK_SIZE=512
export VLLM_MOONCAKE_TRACE_SESSION_HEADER=x-session-id
export VLLM_MOONCAKE_TRACE_MAX_BODY_BYTES=1048576
export VLLM_MOONCAKE_TRACE_INCLUDE_RESPONSE_TEXT=false

vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --device cpu \
  --middleware mooncake_writer.vllm_trace.VLLMMooncakeTraceMiddleware \
  --enable-request-id-headers \
  --enable-force-include-usage
```

If your request `model` is an alias that is not directly loadable as a Hugging Face
tokenizer, set `VLLM_MOONCAKE_TRACE_TOKENIZER=<tokenizer-or-model-name>` so
`hash_only` mode can render the chat template locally.

### Config Knobs

| Variable | Default | Notes |
|---|---|---|
| `VLLM_MOONCAKE_TRACE_ENABLED` | `false` | Global on/off switch |
| `VLLM_MOONCAKE_TRACE_PATH` | `vllm_mooncake_traces.jsonl` | Append-only JSONL output |
| `VLLM_MOONCAKE_TRACE_MODE` | `hash_only` | `raw` preserves chat messages; `hash_only` writes replay-oriented hash blocks |
| `VLLM_MOONCAKE_TRACE_SESSION_HEADER` | `x-session-id` | Optional session ID header to copy into the record |
| `VLLM_MOONCAKE_TRACE_SESSION_BODY_PATH` | unset | Dot path such as `metadata.session_id` |
| `VLLM_MOONCAKE_TRACE_BLOCK_SIZE` | `512` | Passed directly to `MooncakeWriter(block_size=...)` |
| `VLLM_MOONCAKE_TRACE_MAX_BODY_BYTES` | `1048576` | Requests above this size are passed through but not captured |
| `VLLM_MOONCAKE_TRACE_INCLUDE_RESPONSE_TEXT` | `false` | Opt-in raw-mode response text capture |
| `VLLM_MOONCAKE_TRACE_TOKENIZER` | unset | Optional tokenizer override for `hash_only` mode |

### Sample Raw-Mode Line

```json
{"timestamp":0,"captured_at_unix_ms":1712400000000,"session_id":"sess-42","request_id":"req-123","model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","messages":[{"role":"user","content":"Say hello."}],"tools":[{"type":"function","function":{"name":"lookup_weather"}}],"output_length":17,"output_length_source":"usage.completion_tokens"}
```

### Sample Hash-Only Line

```json
{"timestamp":0,"captured_at_unix_ms":1712400000000,"session_id":"sess-42","request_id":"req-123","model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","input_length":48,"output_length":17,"input_length_source":"usage.prompt_tokens","output_length_source":"usage.completion_tokens","hash_ids":[0,1,2]}
```

### Replay With AIPerf

Use a `hash_only` trace file when replaying with the Mooncake trace loader:

```bash
aiperf profile \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --endpoint-type chat \
  --streaming \
  --url http://127.0.0.1:8000/v1 \
  --input-file /tmp/vllm-chat-traces.jsonl \
  --custom-dataset-type mooncake_trace \
  --fixed-schedule
```

### Limitations And Accuracy Notes

- `raw` mode preserves incoming `messages` and optional `tools`, but it intentionally omits `input_length` and `hash_ids`.
- `hash_only` mode uses exact `usage.prompt_tokens` / `usage.completion_tokens` when vLLM includes them, which is why `--enable-force-include-usage` is required.
- Streaming capture preserves the SSE stream verbatim and reads the final usage chunk when present. If usage is missing, `output_length` falls back to local tokenization of streamed assistant text and is labeled with `output_length_source=response_text_tokenization`.
- `hash_only` mode uses this repo's `MooncakeWriter.text_to_hashes()` over the rendered chat prompt. That means the hash IDs follow this repo's existing MooncakeWriter semantics rather than vLLM's internal block hash implementation.
- If the local tokenizer cannot render a chat template, the middleware falls back to hashing a JSON representation of `messages` and `tools` and adds `hash_approximation` to the record.
- v1 writes a single append-only file. It does not include built-in rotation.
- The writer preserves request-arrival ordering within a single process. Multiple server processes can safely append to the same file without corrupting it, but cross-process line ordering is not guaranteed.

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
