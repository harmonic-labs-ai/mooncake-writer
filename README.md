# Mooncake Writer Demo

A cloneable demo showing how to capture [mooncake traces](https://github.com/kvcache-ai/Mooncake) using the `mooncake-writer` library (a wrapper around `aiperf`).

## What are Mooncake Traces?

Mooncake traces represent LLM prompt text as sequences of **hash blocks** — fixed-size chunks of tokens mapped to unique integer IDs via a rolling hash. When two prompts share a common prefix, their hash block sequences share the same leading IDs, directly modelling KV-cache prefix overlap (cache hits). This makes hash blocks a compact, reproducible way to simulate KV-cache behaviour in LLM benchmarks.

## Getting Started

### Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/)

### Setup

```bash
git clone https://github.com/yourusername/mooncake-writer.git
cd mooncake-writer
uv sync --all-extras
```

### Run the Demo

Open the notebook and step through the cells:

```bash
jupyter notebook demo.ipynb
```

## What's in the Notebook

The [`demo.ipynb`](demo.ipynb) notebook walks through:

1. **Initialising** a `MooncakeWriter` with a HuggingFace tokenizer
2. **Converting text to hash blocks** — tokenize, chunk, and hash a single string
3. **Reconstructing text from hashes** — round-trip back to text
4. **Batch operations** — convert multiple texts at once and observe shared prefix hash IDs
5. **Custom block sizes** — adjust the token-per-block granularity

## API Summary

| Method | Description |
|---|---|
| `MooncakeWriter(tokenizer, block_size=512)` | Create a writer with a tokenizer name or instance |
| `text_to_hashes(text)` | Single text to hash ID list |
| `texts_to_hashes(texts)` | Batch texts to hash ID lists |
| `hashes_to_text(hash_ids, input_length)` | Hash IDs back to a single text string |
| `hashes_to_texts(hash_ids_list, input_lengths)` | Batch hash IDs back to text strings |

All methods accept an optional `block_size` override.

## How It Works

1. **Text to Hashes** — Tokenizes text, splits tokens into fixed-size blocks, and assigns consecutive hash IDs using a rolling hash. Shared IDs between sequences represent prefix overlap (cache hits).

2. **Hashes to Text** — Uses a `PromptGenerator` backed by a cached corpus (Shakespeare text) so that the same hash ID always produces the same token block, enabling reproducible text generation from hash IDs.

## License

Apache-2.0
