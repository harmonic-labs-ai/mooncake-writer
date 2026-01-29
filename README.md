# Mooncake Writer

A Python library for converting text to hash blocks and vice versa, wrapping aiperf's mooncake implementation.

## Overview

Mooncake Writer provides a simple interface for converting between text strings and hash block representations. Hash blocks enable efficient representation of text with prefix sharing, which is useful for KV-cache simulation in LLM benchmarking.

## Installation

```bash
pip install mooncake-writer
```

Or install from source using uv (recommended):

```bash
git clone <repository-url>
cd mooncake-writer
uv pip install -e .
```

Or using uv sync:

```bash
git clone <repository-url>
cd mooncake-writer
uv sync
```

## Dependencies

- Python >= 3.11
- aiperf

## Quick Start

```python
from mooncake_writer import MooncakeWriter

# Initialize with a tokenizer model name
writer = MooncakeWriter("gpt2")

# Convert text to hash blocks
text = "Hello, world! This is a test."
hash_ids = writer.text_to_hashes(text)
print(f"Hash IDs: {hash_ids}")

# Convert hash blocks back to text
input_length = 100  # Target length in tokens
reconstructed = writer.hashes_to_text(hash_ids, input_length=input_length)
print(f"Reconstructed: {reconstructed}")
```

## API Reference

### MooncakeWriter

The main class for text-to-hash conversion.

#### `__init__(tokenizer: str | Tokenizer, block_size: int = 512)`

Initialize the MooncakeWriter.

**Parameters:**
- `tokenizer`: Either a model name string (e.g., "gpt2") or a `Tokenizer` instance from aiperf
- `block_size`: Number of tokens per block for hashing. Defaults to 512.

**Raises:**
- `ValueError`: If tokenizer is invalid or block_size is non-positive.

#### `text_to_hashes(text: str, block_size: int | None = None) -> list[int]`

Convert a single text string to hash IDs.

**Parameters:**
- `text`: Input text string to convert
- `block_size`: Number of tokens per block. If None, uses the instance default.

**Returns:**
- List of hash IDs representing the text in blocks.

**Raises:**
- `ValueError`: If text is empty or block_size is invalid.

#### `texts_to_hashes(texts: list[str], block_size: int | None = None) -> list[list[int]]`

Convert multiple text strings to hash ID sequences.

**Parameters:**
- `texts`: List of input text strings to convert
- `block_size`: Number of tokens per block. If None, uses the instance default.

**Returns:**
- List of hash ID sequences, one per input text.

**Raises:**
- `ValueError`: If texts list is empty or contains empty strings, or block_size is invalid.

#### `hashes_to_text(hash_ids: list[int], input_length: int, block_size: int | None = None) -> str`

Convert hash IDs back to a text string.

**Parameters:**
- `hash_ids`: List of hash IDs representing text blocks
- `input_length`: Target input length in tokens
- `block_size`: Number of tokens per block. If None, uses the instance default.

**Returns:**
- Text string reconstructed from the hash IDs.

**Raises:**
- `ValueError`: If hash_ids and input_length are incompatible, or block_size is invalid.

#### `hashes_to_texts(hash_ids_list: list[list[int]], input_lengths: list[int], block_size: int | None = None) -> list[str]`

Convert multiple hash ID sequences back to text strings.

**Parameters:**
- `hash_ids_list`: List of hash ID sequences
- `input_lengths`: Target input lengths (in tokens) for each sequence
- `block_size`: Number of tokens per block. If None, uses the instance default.

**Returns:**
- List of text strings, one per hash ID sequence.

**Raises:**
- `ValueError`: If lists are empty, lengths don't match, or block_size is invalid.

#### `block_size: int` (property)

Get the default block size.

## Examples

### Single Text Conversion

```python
from mooncake_writer import MooncakeWriter

writer = MooncakeWriter("gpt2")

# Convert text to hashes
text = "The quick brown fox jumps over the lazy dog."
hash_ids = writer.text_to_hashes(text)
print(f"Original text: {text}")
print(f"Hash IDs: {hash_ids}")

# Convert back to text
input_length = len(writer._tokenizer.encode(text))
reconstructed = writer.hashes_to_text(hash_ids, input_length=input_length)
print(f"Reconstructed: {reconstructed}")
```

### Batch Operations

```python
from mooncake_writer import MooncakeWriter

writer = MooncakeWriter("gpt2")

# Convert multiple texts to hashes
texts = [
    "First document content here.",
    "Second document with different text.",
    "Third document for processing.",
]

hash_sequences = writer.texts_to_hashes(texts)
print(f"Converted {len(texts)} texts to hash sequences")

# Convert back to texts
input_lengths = [len(writer._tokenizer.encode(text)) for text in texts]
reconstructed_texts = writer.hashes_to_texts(hash_sequences, input_lengths)

for original, reconstructed in zip(texts, reconstructed_texts, strict=True):
    print(f"Original: {original}")
    print(f"Reconstructed: {reconstructed}\n")
```

### Custom Block Size

```python
from mooncake_writer import MooncakeWriter

# Initialize with custom block size
writer = MooncakeWriter("gpt2", block_size=256)

# Use default block size
hash_ids = writer.text_to_hashes("Some text")

# Override block size for specific operation
hash_ids_custom = writer.text_to_hashes("Some text", block_size=128)
```

### Using Custom Tokenizer

```python
from aiperf.common.tokenizer import Tokenizer
from mooncake_writer import MooncakeWriter

# Create a custom tokenizer
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

# Use it with MooncakeWriter
writer = MooncakeWriter(tokenizer)
hash_ids = writer.text_to_hashes("Hello, world!")
```

## How It Works

Mooncake Writer wraps aiperf's mooncake implementation, which:

1. **Text to Hashes**: Tokenizes text, splits into blocks, and generates consecutive hash IDs using a rolling hash algorithm. Shared hash IDs between texts represent prefix overlap (cache hits).

2. **Hashes to Text**: Uses a PromptGenerator with a cached corpus (Shakespeare text) to ensure the same hash ID always produces the same token block, enabling reproducible text generation from hash IDs.

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
