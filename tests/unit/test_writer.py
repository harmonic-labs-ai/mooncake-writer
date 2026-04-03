"""Tests for MooncakeWriter class."""

import json
from pathlib import Path

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

    # ============================================================================
    # Long Context Tests
    # ============================================================================

    def test_long_context_produces_many_blocks(self, mock_tokenizer) -> None:
        """Test that a long text is split into the expected number of blocks."""
        writer = MooncakeWriter(mock_tokenizer, block_size=64)
        text = "x" * 1280  # 1280 tokens / 64 = 20 blocks

        hash_ids = writer.text_to_hashes(text)

        assert len(hash_ids) == 20
        assert all(isinstance(h, int) for h in hash_ids)

    def test_long_context_all_hash_ids_unique(self, mock_tokenizer) -> None:
        """Each block in a single sequence should get a unique hash ID."""
        writer = MooncakeWriter(mock_tokenizer, block_size=32)
        text = "x" * 640  # 640 / 32 = 20 blocks

        hash_ids = writer.text_to_hashes(text)

        assert len(hash_ids) == len(set(hash_ids)), (
            "Expected all hash IDs to be unique within a single sequence"
        )

    def test_long_context_deterministic(self, mock_tokenizer) -> None:
        """Hashing the same long text twice on the same writer produces identical IDs."""
        writer = MooncakeWriter(mock_tokenizer, block_size=64)
        text = "x" * 1280

        ids_first = writer.text_to_hashes(text)
        ids_second = writer.text_to_hashes(text)

        assert ids_first == ids_second

    def test_long_context_partial_last_block(self, mock_tokenizer) -> None:
        """A text whose length isn't a multiple of block_size still hashes correctly."""
        writer = MooncakeWriter(mock_tokenizer, block_size=64)
        text = "x" * 700  # 700 / 64 = 10 full blocks + 1 partial (60 tokens)

        hash_ids = writer.text_to_hashes(text)

        assert len(hash_ids) == 11

    # ============================================================================
    # Cross-Call Prefix Sharing Tests
    # ============================================================================

    def test_cross_call_prefix_sharing(self, mock_tokenizer) -> None:
        """Texts sharing a token prefix produce the same leading hash IDs across calls.

        The mock tokenizer maps text to [0..len-1], so two texts of length 1000
        and 1050 share the first 1000 tokens.  With block_size=100 the first
        10 blocks are identical.
        """
        writer = MooncakeWriter(mock_tokenizer, block_size=100)
        text_a = "x" * 1000  # 10 blocks
        text_b = "x" * 1050  # 10 full shared blocks + 1 partial

        ids_a = writer.text_to_hashes(text_a)
        ids_b = writer.text_to_hashes(text_b)

        assert len(ids_a) == 10
        assert len(ids_b) == 11
        assert ids_a == ids_b[:10], (
            "First 10 blocks share the same tokens and should have identical hash IDs"
        )

    def test_cross_call_divergence(self, mock_tokenizer) -> None:
        """Texts with different lengths produce different hash chains after divergence.

        'x' * 200 → tokens [0..199], blocks [[0..99], [100..199]]
        'x' * 300 → tokens [0..299], blocks [[0..99], [100..199], [200..299]]

        The first two blocks are identical so their IDs match; the third block
        exists only in the longer text so it has a new ID.
        """
        writer = MooncakeWriter(mock_tokenizer, block_size=100)

        ids_short = writer.text_to_hashes("x" * 200)
        ids_long = writer.text_to_hashes("x" * 300)

        assert ids_short == ids_long[:2]
        assert len(ids_long) == 3
        assert ids_long[2] not in ids_short

    def test_cross_call_no_false_sharing(self, mock_tokenizer) -> None:
        """Texts with different token content produce different hash IDs.

        With the mock tokenizer, texts shorter than block_size produce a single
        block whose content depends on length: [0..49] vs [0..69].
        """
        writer = MooncakeWriter(mock_tokenizer, block_size=100)

        ids_a = writer.text_to_hashes("x" * 50)   # 1 block of [0..49]
        ids_b = writer.text_to_hashes("x" * 70)   # 1 block of [0..69]

        assert ids_a != ids_b, (
            "Blocks with different token content should produce different hash IDs"
        )

    # ============================================================================
    # Reset Hashes Tests
    # ============================================================================

    def test_reset_hashes_clears_id_space(self, mock_tokenizer) -> None:
        """After reset_hashes(), IDs start from 0 again."""
        writer = MooncakeWriter(mock_tokenizer, block_size=100)

        writer.text_to_hashes("x" * 500)
        stats_before = writer.hasher.get_stats()
        assert stats_before["total_hashes"] > 0

        writer.reset_hashes()

        stats_after = writer.hasher.get_stats()
        assert stats_after["total_hashes"] == 0
        assert stats_after["max_id"] == -1

    def test_reset_hashes_does_not_affect_traces(self, mock_tokenizer) -> None:
        """reset_hashes() leaves already-captured traces intact."""
        writer = MooncakeWriter(mock_tokenizer, block_size=100)
        writer.capture("x" * 200, timestamp_ms=1000)

        assert len(writer.traces) == 1

        writer.reset_hashes()

        assert len(writer.traces) == 1

    # ============================================================================
    # Capture / Trace Tests
    # ============================================================================

    def test_capture_returns_record(self, mock_tokenizer) -> None:
        """capture() returns a dict with the expected keys."""
        writer = MooncakeWriter(mock_tokenizer, block_size=100)
        record = writer.capture("x" * 200, timestamp_ms=42, output_length=10)

        assert record["timestamp"] == 42
        assert record["input_length"] == 200
        assert isinstance(record["hash_ids"], list)
        assert record["output_length"] == 10

    def test_capture_omits_output_length_when_none(self, mock_tokenizer) -> None:
        """output_length is excluded from the record when not provided."""
        writer = MooncakeWriter(mock_tokenizer, block_size=100)
        record = writer.capture("x" * 100, timestamp_ms=0)

        assert "output_length" not in record

    def test_capture_accumulates_traces(self, mock_tokenizer) -> None:
        """Each capture() call appends to the internal trace buffer."""
        writer = MooncakeWriter(mock_tokenizer, block_size=100)

        writer.capture("x" * 100, timestamp_ms=0)
        writer.capture("x" * 200, timestamp_ms=100)
        writer.capture("x" * 300, timestamp_ms=200)

        assert len(writer.traces) == 3

    def test_clear_trace(self, mock_tokenizer) -> None:
        """clear_trace() empties the buffer but preserves hash state."""
        writer = MooncakeWriter(mock_tokenizer, block_size=100)
        writer.capture("x" * 200, timestamp_ms=0)

        stats_before = writer.hasher.get_stats()
        writer.clear_trace()

        assert len(writer.traces) == 0
        assert writer.hasher.get_stats() == stats_before

    def test_write_trace_jsonl(self, mock_tokenizer, tmp_path: Path) -> None:
        """write_trace() writes valid JSONL matching aiperf MooncakeTrace schema."""
        writer = MooncakeWriter(mock_tokenizer, block_size=100)
        writer.capture("x" * 200, timestamp_ms=1000, output_length=50)
        writer.capture("x" * 300, timestamp_ms=2000)

        out = tmp_path / "trace.jsonl"
        count = writer.write_trace(out)

        assert count == 2

        lines = out.read_text().strip().splitlines()
        assert len(lines) == 2

        rec0 = json.loads(lines[0])
        assert rec0["timestamp"] == 1000
        assert rec0["input_length"] == 200
        assert isinstance(rec0["hash_ids"], list)
        assert rec0["output_length"] == 50

        rec1 = json.loads(lines[1])
        assert rec1["timestamp"] == 2000
        assert "output_length" not in rec1

    def test_traces_returns_copy(self, mock_tokenizer) -> None:
        """The traces property returns a copy, not a reference to the internal list."""
        writer = MooncakeWriter(mock_tokenizer, block_size=100)
        writer.capture("x" * 100, timestamp_ms=0)

        traces_copy = writer.traces
        traces_copy.clear()

        assert len(writer.traces) == 1
