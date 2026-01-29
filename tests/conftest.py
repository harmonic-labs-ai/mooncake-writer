"""Pytest configuration and shared fixtures."""

from unittest.mock import MagicMock

import pytest

from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.generator.prompt import PromptGenerator


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock(spec=Tokenizer)
    # Simple tokenizer: 1 token per character for deterministic testing
    tokenizer.encode = lambda text, **kwargs: list(range(len(text)))
    tokenizer.decode = lambda token_ids, **kwargs: "".join(
        chr(ord("a") + (t % 26)) for t in token_ids
    )
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.block_separation_token_id = 1
    return tokenizer


@pytest.fixture
def mock_prompt_generator(mock_tokenizer: MagicMock) -> MagicMock:
    """Create a mock prompt generator for testing."""
    generator = MagicMock(spec=PromptGenerator)
    generator.generate = MagicMock(return_value="generated text from hash ids")
    generator.tokenizer = mock_tokenizer
    return generator


@pytest.fixture
def real_tokenizer() -> Tokenizer:
    """Create a real tokenizer instance for integration tests."""
    return Tokenizer.from_pretrained("gpt2")
