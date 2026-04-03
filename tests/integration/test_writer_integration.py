"""Integration tests for MooncakeWriter using a real GPT-2 tokenizer."""

import json
from pathlib import Path

import pytest

from mooncake_writer import MooncakeWriter

BLOCK_SIZE = 32

LONG_TEXT = (
    "In the beginning, the universe was created. This has made a lot of "
    "people very angry and been widely regarded as a bad move. Many were "
    "increasingly of the opinion that they'd all made a big mistake in "
    "coming down from the trees in the first place. And some said that "
    "even the trees had been a bad move, and that no one should ever have "
    "left the oceans. The ships hung in the sky in much the same way that "
    "bricks don't. It is a well-known fact that those people who must want "
    "to rule people are, ipso facto, those least suited to do it. Anyone "
    "who is capable of getting themselves made President should on no "
    "account be allowed to do the job. Time is an illusion. Lunchtime "
    "doubly so. The Answer to the Great Question of Life, the Universe "
    "and Everything is Forty-two."
)

SHARED_PREFIX = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "The five boxing wizards jump quickly. "
    "Sphinx of black quartz, judge my vow. "
)

UNRELATED_TEXT = (
    "Quantum computing leverages superposition and entanglement to perform "
    "calculations that would be infeasible on classical hardware. Recent "
    "advances in error correction have brought fault-tolerant systems closer "
    "to reality, with superconducting qubits achieving coherence times that "
    "enable meaningful circuit depths for combinatorial optimisation problems."
)


@pytest.fixture
def writer() -> MooncakeWriter:
    return MooncakeWriter("gpt2", block_size=BLOCK_SIZE)


@pytest.mark.integration
class TestLongContext:
    """Validate hashing behaviour with real tokenization and many blocks."""

    def test_long_context_many_blocks(self, writer: MooncakeWriter) -> None:
        """A long text produces many blocks, each with a unique hash ID."""
        hash_ids = writer.text_to_hashes(LONG_TEXT)
        expected_blocks = -(
            -len(writer.tokenizer.encode(LONG_TEXT)) // BLOCK_SIZE
        )

        assert len(hash_ids) == expected_blocks
        assert len(hash_ids) >= 5, "Sanity: text should produce at least 5 blocks"
        assert len(hash_ids) == len(set(hash_ids)), (
            "All hash IDs should be unique within a single sequence"
        )

    def test_deterministic_across_calls(self, writer: MooncakeWriter) -> None:
        """Same text hashed twice on the same writer returns identical IDs."""
        ids_first = writer.text_to_hashes(LONG_TEXT)
        ids_second = writer.text_to_hashes(LONG_TEXT)

        assert ids_first == ids_second


@pytest.mark.integration
class TestPrefixSharing:
    """Validate cross-call prefix sharing with real English text."""

    def test_prefix_sharing_across_calls(self, writer: MooncakeWriter) -> None:
        """Texts sharing a long prefix get the same leading hash IDs."""
        text_a = SHARED_PREFIX + "The dog barked loudly at the mailman."
        text_b = SHARED_PREFIX + "A cat slept peacefully on the windowsill."

        ids_a = writer.text_to_hashes(text_a)
        ids_b = writer.text_to_hashes(text_b)

        prefix_tokens_a = writer.tokenizer.encode(SHARED_PREFIX)
        shared_blocks = len(prefix_tokens_a) // BLOCK_SIZE

        assert shared_blocks >= 1, "Sanity: shared prefix should span at least 1 block"
        assert ids_a[:shared_blocks] == ids_b[:shared_blocks], (
            f"First {shared_blocks} blocks should share hash IDs"
        )
        assert ids_a != ids_b, "Full sequences should differ after the shared prefix"

    def test_prefix_sharing_different_content(self, writer: MooncakeWriter) -> None:
        """Completely unrelated texts share no hash IDs."""
        ids_a = writer.text_to_hashes(SHARED_PREFIX)
        ids_b = writer.text_to_hashes(UNRELATED_TEXT)

        shared = set(ids_a) & set(ids_b)
        assert shared == set(), f"Expected no shared hash IDs, got {shared}"


@pytest.mark.integration
class TestTraceCapture:
    """Validate capture and JSONL output with real text."""

    def test_capture_with_real_text(self, writer: MooncakeWriter) -> None:
        """capture() records correct input_length from real tokenization."""
        record = writer.capture(LONG_TEXT, timestamp_ms=1000, output_length=64)

        expected_length = len(writer.tokenizer.encode(LONG_TEXT))
        assert record["input_length"] == expected_length
        assert record["timestamp"] == 1000
        assert record["output_length"] == 64
        assert len(record["hash_ids"]) >= 5

    def test_write_trace_round_trip(
        self, writer: MooncakeWriter, tmp_path: Path
    ) -> None:
        """Captured traces round-trip through JSONL correctly."""
        texts = [SHARED_PREFIX, UNRELATED_TEXT, LONG_TEXT]
        for i, text in enumerate(texts):
            writer.capture(text, timestamp_ms=i * 1000, output_length=20 + i)

        out = tmp_path / "trace.jsonl"
        count = writer.write_trace(out)

        assert count == 3
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 3

        for i, (line, text) in enumerate(zip(lines, texts)):
            rec = json.loads(line)
            assert rec["timestamp"] == i * 1000
            assert rec["input_length"] == len(writer.tokenizer.encode(text))
            assert rec["output_length"] == 20 + i
            assert isinstance(rec["hash_ids"], list)
            assert len(rec["hash_ids"]) >= 1
