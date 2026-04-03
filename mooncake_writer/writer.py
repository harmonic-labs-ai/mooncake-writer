"""Core MooncakeWriter class for text-to-hash conversion and trace capture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aiperf.common.config.config_defaults import InputTokensDefaults
from aiperf.common.config.prompt_config import PromptConfig
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.generator.prompt import PromptGenerator
from mooncake_writer.rolling_hasher import RollingHasher, hashes_to_texts


class MooncakeWriter:
    """Converts text to hash blocks, captures traces, and writes aiperf-compatible JSONL.

    Maintains a persistent ``RollingHasher`` so that hash state (the hash-to-ID
    mapping) is preserved across calls.  Two texts that share a token prefix
    will receive the same leading hash IDs regardless of whether they are
    processed in the same call or separate calls.

    The ``capture`` / ``write_trace`` workflow lets you record timestamped
    trace records and flush them to a JSONL file that aiperf can load as a
    ``MooncakeTrace`` dataset.

    Example:
        >>> import time
        >>> writer = MooncakeWriter("gpt2")
        >>> writer.capture("Hello, world!", timestamp_ms=int(time.time() * 1000), output_length=20)
        >>> writer.capture("Hello, universe!", timestamp_ms=int(time.time() * 1000), output_length=30)
        >>> writer.write_trace("trace.jsonl")
        2
    """

    def __init__(
        self,
        tokenizer: str | Tokenizer,
        block_size: int = InputTokensDefaults.BLOCK_SIZE,
    ) -> None:
        """Initialize the MooncakeWriter.

        Args:
            tokenizer: Either a model name string (e.g., "gpt2") or a Tokenizer instance.
                If a string is provided, a Tokenizer will be created using from_pretrained.
            block_size: Number of tokens per block for hashing. Defaults to 512.

        Raises:
            ValueError: If tokenizer is invalid or block_size is non-positive.
        """
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        if isinstance(tokenizer, str):
            self._tokenizer = Tokenizer.from_pretrained(tokenizer)
        elif isinstance(tokenizer, Tokenizer):
            self._tokenizer = tokenizer
        else:
            raise ValueError(
                f"tokenizer must be a string or Tokenizer instance, got {type(tokenizer)}"
            )

        prompt_config = PromptConfig()
        self._prompt_generator = PromptGenerator(
            config=prompt_config, tokenizer=self._tokenizer
        )

        self._block_size = block_size
        self._hasher = RollingHasher(block_size=block_size)
        self._traces: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    def text_to_hashes(
        self, text: str, block_size: int | None = None
    ) -> list[int]:
        """Convert a single text string to hash IDs.

        Args:
            text: Input text string to convert.
            block_size: Number of tokens per block. If None, uses the instance default.

        Returns:
            List of hash IDs representing the text in blocks.

        Raises:
            ValueError: If text is empty or block_size is invalid.
        """
        if not text:
            raise ValueError("text cannot be empty")

        size = block_size if block_size is not None else self._block_size
        if size <= 0:
            raise ValueError(f"block_size must be positive, got {size}")

        tokens = self._tokenizer.encode(text)
        blocks = [tokens[i : i + size] for i in range(0, len(tokens), size)]
        return self._hasher.hash_token_blocks(blocks) if blocks else []

    def texts_to_hashes(
        self, texts: list[str], block_size: int | None = None
    ) -> list[list[int]]:
        """Convert multiple text strings to hash ID sequences.

        Args:
            texts: List of input text strings to convert.
            block_size: Number of tokens per block. If None, uses the instance default.

        Returns:
            List of hash ID sequences, one per input text.

        Raises:
            ValueError: If texts list is empty or contains empty strings, or block_size is invalid.
        """
        if not texts:
            raise ValueError("texts list cannot be empty")

        if any(not text for text in texts):
            raise ValueError("texts list cannot contain empty strings")

        size = block_size if block_size is not None else self._block_size
        if size <= 0:
            raise ValueError(f"block_size must be positive, got {size}")

        results: list[list[int]] = []
        for text in texts:
            tokens = self._tokenizer.encode(text)
            blocks = [tokens[i : i + size] for i in range(0, len(tokens), size)]
            results.append(self._hasher.hash_token_blocks(blocks) if blocks else [])
        return results

    # ------------------------------------------------------------------
    # Reverse mapping (hashes -> text)
    # ------------------------------------------------------------------

    def hashes_to_text(
        self,
        hash_ids: list[int],
        input_length: int,
        block_size: int | None = None,
    ) -> str:
        """Convert hash IDs back to a text string.

        Args:
            hash_ids: List of hash IDs representing text blocks.
            input_length: Target input length in tokens.
            block_size: Number of tokens per block. If None, uses the instance default.

        Returns:
            Text string reconstructed from the hash IDs.

        Raises:
            ValueError: If hash_ids and input_length are incompatible, or block_size is invalid.
        """
        size = block_size if block_size is not None else self._block_size
        if size <= 0:
            raise ValueError(f"block_size must be positive, got {size}")

        result = hashes_to_texts(
            self._prompt_generator,
            [hash_ids],
            [input_length],
            block_size=size,
        )
        return result[0] if result else ""

    def hashes_to_texts(
        self,
        hash_ids_list: list[list[int]],
        input_lengths: list[int],
        block_size: int | None = None,
    ) -> list[str]:
        """Convert multiple hash ID sequences back to text strings.

        Args:
            hash_ids_list: List of hash ID sequences.
            input_lengths: Target input lengths (in tokens) for each sequence.
            block_size: Number of tokens per block. If None, uses the instance default.

        Returns:
            List of text strings, one per hash ID sequence.

        Raises:
            ValueError: If lists are empty, lengths don't match, or block_size is invalid.
        """
        if not hash_ids_list:
            raise ValueError("hash_ids_list cannot be empty")

        if not input_lengths:
            raise ValueError("input_lengths cannot be empty")

        if len(hash_ids_list) != len(input_lengths):
            raise ValueError(
                f"hash_ids_list and input_lengths must have the same length, "
                f"got {len(hash_ids_list)} and {len(input_lengths)}"
            )

        size = block_size if block_size is not None else self._block_size
        if size <= 0:
            raise ValueError(f"block_size must be positive, got {size}")

        return hashes_to_texts(
            self._prompt_generator,
            hash_ids_list,
            input_lengths,
            block_size=size,
        )

    # ------------------------------------------------------------------
    # Trace capture
    # ------------------------------------------------------------------

    def capture(
        self,
        text: str,
        timestamp_ms: int,
        output_length: int | None = None,
    ) -> dict[str, Any]:
        """Hash a text and record a MooncakeTrace-compatible trace record.

        Args:
            text: Input text string (the prompt).
            timestamp_ms: Request arrival time in milliseconds.  The caller is
                responsible for capturing this at the appropriate moment
                (e.g. when the request is first received by the proxy).
            output_length: Max tokens to generate for this request. Omitted from
                the trace record when ``None``.

        Returns:
            The trace record dict that was appended to the internal buffer.
        """
        hash_ids = self.text_to_hashes(text)
        input_length = len(self._tokenizer.encode(text))

        record: dict[str, Any] = {
            "timestamp": timestamp_ms,
            "input_length": input_length,
            "hash_ids": hash_ids,
        }
        if output_length is not None:
            record["output_length"] = output_length

        self._traces.append(record)
        return record

    def write_trace(self, path: str | Path) -> int:
        """Write captured traces to a JSONL file compatible with aiperf's MooncakeTrace loader.

        Args:
            path: Destination file path. Parent directories must exist.

        Returns:
            Number of records written.
        """
        path = Path(path)
        with path.open("w") as f:
            for record in self._traces:
                f.write(json.dumps(record) + "\n")
        return len(self._traces)

    def clear_trace(self) -> None:
        """Clear the internal trace buffer without resetting hash state."""
        self._traces.clear()

    # ------------------------------------------------------------------
    # Hash state management
    # ------------------------------------------------------------------

    def reset_hashes(self) -> None:
        """Reset the internal hasher, clearing all hash-to-ID mappings.

        Previously captured traces are not affected.
        """
        self._hasher = RollingHasher(block_size=self._block_size)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def block_size(self) -> int:
        """Get the default block size."""
        return self._block_size

    @property
    def tokenizer(self) -> Tokenizer:
        """Get the tokenizer instance."""
        return self._tokenizer

    @property
    def hasher(self) -> RollingHasher:
        """Get the underlying RollingHasher instance."""
        return self._hasher

    @property
    def traces(self) -> list[dict[str, Any]]:
        """Get a copy of the captured trace records."""
        return list(self._traces)
