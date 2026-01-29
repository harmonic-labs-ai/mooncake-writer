"""Tests for MooncakeWriter class."""

import pytest

from mooncake_writer import MooncakeWriter


class TestMooncakeWriter:
    """Tests for MooncakeWriter class."""

    # ============================================================================
    # Initialization Tests
    # ============================================================================

    def test_initialization_with_string_tokenizer(self, monkeypatch, mock_tokenizer) -> None:
        """Test initialization with tokenizer string."""
        # Mock Tokenizer.from_pretrained to return our mock
        from mooncake_writer import writer as writer_module

        monkeypatch.setattr(
            writer_module.Tokenizer,
            "from_pretrained",
            lambda *args, **kwargs: mock_tokenizer,
        )

        writer = MooncakeWriter("gpt2")
        assert writer.tokenizer == mock_tokenizer
        assert writer.block_size == 512

    def test_initialization_with_tokenizer_instance(self, mock_tokenizer) -> None:
        """Test initialization with Tokenizer instance."""
        writer = MooncakeWriter(mock_tokenizer)
        assert writer.tokenizer == mock_tokenizer
        assert writer.block_size == 512

    def test_initialization_custom_block_size(self, mock_tokenizer) -> None:
        """Test initialization with custom block size."""
        writer = MooncakeWriter(mock_tokenizer, block_size=256)
        assert writer.block_size == 256

    def test_initialization_invalid_block_size(self, mock_tokenizer) -> None:
        """Test initialization with invalid block size raises ValueError."""
        with pytest.raises(ValueError, match="block_size must be positive"):
            MooncakeWriter(mock_tokenizer, block_size=0)

        with pytest.raises(ValueError, match="block_size must be positive"):
            MooncakeWriter(mock_tokenizer, block_size=-1)

    def test_initialization_invalid_tokenizer_type(self) -> None:
        """Test initialization with invalid tokenizer type raises ValueError."""
        with pytest.raises(ValueError, match="tokenizer must be a string or Tokenizer"):
            MooncakeWriter(123)  # type: ignore

    # ============================================================================
    # Text to Hashes Tests
    # ============================================================================

    def test_text_to_hashes_single(self, mock_tokenizer) -> None:
        """Test converting single text to hashes."""
        writer = MooncakeWriter(mock_tokenizer, block_size=10)
        text = "a" * 20  # 20 tokens = 2 blocks

        hash_ids = writer.text_to_hashes(text)

        assert len(hash_ids) == 2
        assert all(isinstance(h, int) for h in hash_ids)

    def test_text_to_hashes_custom_block_size(self, mock_tokenizer) -> None:
        """Test converting text with custom block size."""
        writer = MooncakeWriter(mock_tokenizer, block_size=10)
        text = "a" * 30  # 30 tokens

        hash_ids = writer.text_to_hashes(text, block_size=15)

        assert len(hash_ids) == 2  # 30 / 15 = 2 blocks

    def test_text_to_hashes_empty_text(self, mock_tokenizer) -> None:
        """Test converting empty text raises ValueError."""
        writer = MooncakeWriter(mock_tokenizer)

        with pytest.raises(ValueError, match="text cannot be empty"):
            writer.text_to_hashes("")

    def test_texts_to_hashes_multiple(self, mock_tokenizer) -> None:
        """Test converting multiple texts to hashes."""
        writer = MooncakeWriter(mock_tokenizer, block_size=10)
        texts = ["a" * 20, "b" * 30]

        hash_sequences = writer.texts_to_hashes(texts)

        assert len(hash_sequences) == 2
        assert len(hash_sequences[0]) == 2  # 20 / 10 = 2 blocks
        assert len(hash_sequences[1]) == 3  # 30 / 10 = 3 blocks

    def test_texts_to_hashes_empty_list(self, mock_tokenizer) -> None:
        """Test converting empty texts list raises ValueError."""
        writer = MooncakeWriter(mock_tokenizer)

        with pytest.raises(ValueError, match="texts list cannot be empty"):
            writer.texts_to_hashes([])

    def test_texts_to_hashes_contains_empty_string(self, mock_tokenizer) -> None:
        """Test converting texts with empty string raises ValueError."""
        writer = MooncakeWriter(mock_tokenizer)

        with pytest.raises(ValueError, match="texts list cannot contain empty strings"):
            writer.texts_to_hashes(["hello", ""])

    # ============================================================================
    # Hashes to Text Tests
    # ============================================================================

    def test_hashes_to_text_single(
        self, mock_tokenizer, mock_prompt_generator, monkeypatch
    ) -> None:
        """Test converting single hash sequence to text."""
        # Mock the PromptGenerator creation
        from mooncake_writer import writer as writer_module

        def mock_init(self, config, tokenizer):
            self.generate = mock_prompt_generator.generate

        monkeypatch.setattr(writer_module.PromptGenerator, "__init__", mock_init)

        writer = MooncakeWriter(mock_tokenizer)
        hash_ids = [1, 2, 3]
        input_length = 100

        result = writer.hashes_to_text(hash_ids, input_length)

        assert isinstance(result, str)
        mock_prompt_generator.generate.assert_called_once()

    def test_hashes_to_texts_multiple(
        self, mock_tokenizer, mock_prompt_generator, monkeypatch
    ) -> None:
        """Test converting multiple hash sequences to texts."""
        from mooncake_writer import writer as writer_module

        def mock_init(self, config, tokenizer):
            self.generate = mock_prompt_generator.generate

        monkeypatch.setattr(writer_module.PromptGenerator, "__init__", mock_init)

        writer = MooncakeWriter(mock_tokenizer)
        hash_ids_list = [[1, 2], [3, 4, 5]]
        input_lengths = [100, 150]

        result = writer.hashes_to_texts(hash_ids_list, input_lengths)

        assert len(result) == 2
        assert mock_prompt_generator.generate.call_count == 2

    def test_hashes_to_texts_mismatched_lengths(self, mock_tokenizer) -> None:
        """Test converting with mismatched lengths raises ValueError."""
        writer = MooncakeWriter(mock_tokenizer)
        hash_ids_list = [[1, 2], [3, 4]]
        input_lengths = [100]  # Mismatched length

        with pytest.raises(
            ValueError, match="hash_ids_list and input_lengths must have the same length"
        ):
            writer.hashes_to_texts(hash_ids_list, input_lengths)

    def test_hashes_to_texts_empty_list(self, mock_tokenizer) -> None:
        """Test converting empty hash_ids_list raises ValueError."""
        writer = MooncakeWriter(mock_tokenizer)

        with pytest.raises(ValueError, match="hash_ids_list cannot be empty"):
            writer.hashes_to_texts([], [100])

        with pytest.raises(ValueError, match="input_lengths cannot be empty"):
            writer.hashes_to_texts([[1, 2]], [])

    # ============================================================================
    # Property Tests
    # ============================================================================

    def test_block_size_property(self, mock_tokenizer) -> None:
        """Test block_size property."""
        writer = MooncakeWriter(mock_tokenizer, block_size=256)
        assert writer.block_size == 256

    def test_tokenizer_property(self, mock_tokenizer) -> None:
        """Test tokenizer property."""
        writer = MooncakeWriter(mock_tokenizer)
        assert writer.tokenizer == mock_tokenizer
