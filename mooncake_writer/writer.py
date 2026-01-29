"""Core MooncakeWriter class for text-to-hash conversion."""

from typing import TYPE_CHECKING

from aiperf.common.config.config_defaults import InputTokensDefaults
from aiperf.common.config.prompt_config import PromptConfig
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.generator.prompt import PromptGenerator
from mooncake_writer.rolling_hasher import hashes_to_texts, texts_to_hashes

if TYPE_CHECKING:
    pass


class MooncakeWriter:
    """A class for converting text to hash blocks and vice versa.

    This class wraps aiperf's mooncake implementation to provide a simple
    interface for converting between text strings and hash block representations.
    Hash blocks enable efficient representation of text with prefix sharing for
    KV-cache simulation.

    Example:
        >>> writer = MooncakeWriter("gpt2")
        >>> hash_ids = writer.text_to_hashes("Hello, world!")
        >>> text = writer.hashes_to_text(hash_ids, input_length=100)
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

        # Initialize tokenizer
        if isinstance(tokenizer, str):
            self._tokenizer = Tokenizer.from_pretrained(tokenizer)
        elif isinstance(tokenizer, Tokenizer):
            self._tokenizer = tokenizer
        else:
            raise ValueError(
                f"tokenizer must be a string or Tokenizer instance, got {type(tokenizer)}"
            )

        # Initialize PromptGenerator with default config
        prompt_config = PromptConfig()
        self._prompt_generator = PromptGenerator(
            config=prompt_config, tokenizer=self._tokenizer
        )

        self._block_size = block_size

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

        result = texts_to_hashes(self._tokenizer, [text], block_size=size)
        return result[0] if result else []

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

        return texts_to_hashes(self._tokenizer, texts, block_size=size)

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

    @property
    def block_size(self) -> int:
        """Get the default block size."""
        return self._block_size

    @property
    def tokenizer(self) -> Tokenizer:
        """Get the tokenizer instance."""
        return self._tokenizer
