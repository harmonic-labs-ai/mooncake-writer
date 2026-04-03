"""Tests for RollingHasher."""

import pytest

from mooncake_writer.rolling_hasher import (
    RollingHasher,
    hashes_to_texts,
    texts_to_hashes,
)


class TestRollingHasher:
    """Tests for RollingHasher class."""

    # ============================================================================
    # Initialization Tests
    # ============================================================================

    def test_initialization_default(self) -> None:
        """Test RollingHasher initialization with defaults."""
        hasher = RollingHasher()
        assert hasher.block_size == 512
        stats = hasher.get_stats()
        assert stats["total_hashes"] == 0
        assert stats["max_id"] == -1

    def test_initialization_custom_block_size(self) -> None:
        """Test RollingHasher initialization with custom block size."""
        hasher = RollingHasher(block_size=256)
        assert hasher.block_size == 256

    # ============================================================================
    # Hash Generation Tests
    # ============================================================================

    def test_hash_single_block(self) -> None:
        """Test hashing a single block."""
        hasher = RollingHasher()
        hash_ids = hasher.hash_blocks(["hello"])
        assert len(hash_ids) == 1
        assert hash_ids[0] == 0

    def test_hash_multiple_blocks(self) -> None:
        """Test hashing multiple blocks."""
        hasher = RollingHasher()
        hash_ids = hasher.hash_blocks(["hello", "world", "test"])
        assert len(hash_ids) == 3
        assert all(isinstance(h, int) for h in hash_ids)

    def test_hash_empty_list(self) -> None:
        """Test hashing empty list returns empty list."""
        hasher = RollingHasher()
        hash_ids = hasher.hash_blocks([])
        assert hash_ids == []

    @pytest.mark.parametrize(
        "blocks,expected_count",
        [
            (["a"], 1),
            (["a", "b"], 2),
            (["a", "b", "c", "d", "e"], 5),
        ],
    )
    def test_hash_sequence_lengths(
        self, blocks: list[str], expected_count: int
    ) -> None:
        """Test that output length matches input length."""
        hasher = RollingHasher()
        hash_ids = hasher.hash_blocks(blocks)
        assert len(hash_ids) == expected_count

    # ============================================================================
    # Rolling Hash State Tests
    # ============================================================================

    def test_rolling_hash_context_matters(self) -> None:
        """Test that rolling hash context affects the hash ID."""
        hasher1 = RollingHasher()
        hash_ids1 = hasher1.hash_blocks(["a", "b"])

        hasher2 = RollingHasher()
        hash_ids2 = hasher2.hash_blocks(["a"])

        # The second "a" in hasher1's sequence is different from hasher2's "a"
        # because it has different context (different previous hash)
        assert len(hash_ids1) == 2
        assert len(hash_ids2) == 1

    def test_reset_clears_state(self) -> None:
        """Test that reset clears the rolling state."""
        hasher = RollingHasher()
        hash_ids1 = hasher.hash_blocks(["a", "b"])

        hasher.reset()

        hash_ids2 = hasher.hash_blocks(["a", "b"])

        # After reset, the same sequence should produce different context-based hashes
        assert len(hash_ids1) == len(hash_ids2)

    # ============================================================================
    # Statistics Tests
    # ============================================================================

    def test_get_stats_counts(self) -> None:
        """Test that statistics accurately count hashes."""
        hasher = RollingHasher()
        hasher.hash_blocks(["a", "b", "c"])

        stats = hasher.get_stats()
        assert stats["total_hashes"] > 0  # Should have seen some hashes
        assert stats["max_id"] >= 0

    # ============================================================================
    # Token Block Hashing Tests
    # ============================================================================

    def test_hash_token_blocks_single(self) -> None:
        """Test hashing a single token block."""
        hasher = RollingHasher()
        blocks = [[1, 2, 3, 4]]
        hash_ids = hasher.hash_token_blocks(blocks)
        assert len(hash_ids) == 1
        assert isinstance(hash_ids[0], int)

    def test_hash_token_blocks_multiple(self) -> None:
        """Test hashing multiple token blocks."""
        hasher = RollingHasher()
        blocks = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        hash_ids = hasher.hash_token_blocks(blocks)
        assert len(hash_ids) == 3
        assert all(isinstance(h, int) for h in hash_ids)

    def test_hash_token_blocks_empty(self) -> None:
        """Test hashing empty token block list."""
        hasher = RollingHasher()
        hash_ids = hasher.hash_token_blocks([])
        assert hash_ids == []

    def test_hash_token_blocks_preserves_prefix_sharing(self) -> None:
        """Test that shared prefixes produce same hash IDs."""
        hasher = RollingHasher()
        # Two sequences with shared prefix
        seq1 = [[1, 2], [3, 4], [5, 6]]
        seq2 = [[1, 2], [3, 4], [7, 8]]

        hash_ids1 = hasher.hash_token_blocks(seq1)
        hasher.reset()
        hash_ids2 = hasher.hash_token_blocks(seq2)

        # First two blocks should have same hash IDs (shared prefix)
        assert hash_ids1[0] == hash_ids2[0]
        assert hash_ids1[1] == hash_ids2[1]


class TestTextsToHashes:
    """Tests for texts_to_hashes module function."""

    def test_texts_to_hashes_single_text(self, mock_tokenizer) -> None:
        """Test converting a single text to hashes."""
        texts = ["a" * 20]  # 20 tokens with block_size=10 = 2 blocks
        result = texts_to_hashes(mock_tokenizer, texts, block_size=10)

        assert len(result) == 1
        assert len(result[0]) == 2  # 2 blocks
        assert all(isinstance(h, int) for h in result[0])

    def test_texts_to_hashes_multiple_texts(self, mock_tokenizer) -> None:
        """Test converting multiple texts to hashes."""
        texts = ["a" * 20, "b" * 30]
        result = texts_to_hashes(mock_tokenizer, texts, block_size=10)

        assert len(result) == 2
        assert len(result[0]) == 2  # 20 tokens / 10 = 2 blocks
        assert len(result[1]) == 3  # 30 tokens / 10 = 3 blocks

    def test_texts_to_hashes_empty_text(self, mock_tokenizer) -> None:
        """Test converting empty text returns empty hash list."""
        texts = [""]
        result = texts_to_hashes(mock_tokenizer, texts, block_size=10)

        assert len(result) == 1
        assert result[0] == []

    def test_texts_to_hashes_shared_prefix(self, mock_tokenizer) -> None:
        """Test that shared token prefixes produce shared hash IDs.

        The mock tokenizer produces [0..N-1] for a text of length N, so
        "a"*20 -> [0..19] (2 blocks of 10) and "a"*25 -> [0..24] (2 full + 1 partial).
        The first block [0..9] is identical; the second block differs
        ([10..19] vs [10..19,20..24] -- wait, both have [10..19] as the full
        second block).  Use "a"*20 (2 blocks) vs "a"*15 (1 full + 1 partial)
        so the second block differs: [10..19] vs [10..14].
        """
        texts = ["a" * 20, "a" * 15]
        result = texts_to_hashes(mock_tokenizer, texts, block_size=10)

        assert len(result) == 2
        assert len(result[0]) == 2  # 20 / 10 = 2 blocks
        assert len(result[1]) == 2  # 15 / 10 = 1 full + 1 partial
        assert result[0][0] == result[1][0], "First block shares tokens [0..9]"
        assert result[0][1] != result[1][1], "Second block differs: [10..19] vs [10..14]"


class TestHashesToTexts:
    """Tests for hashes_to_texts module function."""

    def test_hashes_to_texts_single(self, mock_prompt_generator) -> None:
        """Test converting single hash sequence to text."""
        hash_ids_list = [[1, 2, 3]]
        input_lengths = [100]

        result = hashes_to_texts(
            mock_prompt_generator, hash_ids_list, input_lengths, block_size=64
        )

        assert len(result) == 1
        mock_prompt_generator.generate.assert_called_once()

    def test_hashes_to_texts_multiple(self, mock_prompt_generator) -> None:
        """Test converting multiple hash sequences to texts."""
        hash_ids_list = [[1, 2], [3, 4, 5]]
        input_lengths = [100, 150]

        result = hashes_to_texts(
            mock_prompt_generator, hash_ids_list, input_lengths, block_size=64
        )

        assert len(result) == 2
        assert mock_prompt_generator.generate.call_count == 2

    def test_hashes_to_texts_empty_hash_ids(self, mock_prompt_generator) -> None:
        """Test converting empty hash_ids generates text without hash_ids."""
        hash_ids_list = [[]]
        input_lengths = [100]

        result = hashes_to_texts(
            mock_prompt_generator, hash_ids_list, input_lengths, block_size=64
        )

        assert len(result) == 1
        # Should call generate without hash_ids
        mock_prompt_generator.generate.assert_called_with(mean=100)

    def test_hashes_to_texts_constraint_violation(self, mock_prompt_generator) -> None:
        """Test that constraint violation raises ValueError."""
        # 2 hash_ids * 64 block_size = 128 < 200 input_length
        hash_ids_list = [[1, 2]]
        input_lengths = [200]

        with pytest.raises(ValueError, match="Constraint violation"):
            hashes_to_texts(
                mock_prompt_generator, hash_ids_list, input_lengths, block_size=64
            )
