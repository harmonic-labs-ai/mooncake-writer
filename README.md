# Mooncake Writer Demo

A cloneable demo showing how to capture [Mooncake traces](https://github.com/kvcache-ai/Mooncake) using the `mooncake-writer` library (a small wrapper around [AIPerf](https://github.com/ai-dynamo/aiperf)).

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
