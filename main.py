"""Example usage of mooncake-writer library."""

from mooncake_writer import MooncakeWriter


def main():
    """Demonstrate mooncake-writer functionality."""
    print("Mooncake Writer Example")
    print("=" * 50)

    # Initialize the writer with a tokenizer model
    print("\n1. Initializing MooncakeWriter with 'gpt2' tokenizer...")
    writer = MooncakeWriter("gpt2")

    # Example 1: Convert single text to hashes
    print("\n2. Converting text to hash blocks...")
    text1 = "Hello, world! This is a test of the mooncake writer."
    hash_ids = writer.text_to_hashes(text1)
    print(f"   Text: {text1[:50]}...")
    print(f"   Hash IDs: {hash_ids}")

    # Example 2: Convert hashes back to text
    print("\n3. Converting hash blocks back to text...")
    # Note: input_length should match the tokenized length of the original text
    # For demonstration, we'll use a reasonable estimate
    input_length = len(writer.tokenizer.encode(text1))
    reconstructed = writer.hashes_to_text(hash_ids, input_length=input_length)
    print(f"   Reconstructed text: {reconstructed[:50]}...")

    # Example 3: Batch operations
    print("\n4. Batch operations...")
    texts = [
        "First text to convert.",
        "Second text with different content.",
        "Third text for batch processing.",
    ]
    hash_sequences = writer.texts_to_hashes(texts)
    print(f"   Converted {len(texts)} texts to hash sequences")
    for i, (text, hashes) in enumerate(zip(texts, hash_sequences, strict=True)):
        print(f"   Text {i+1}: {len(hashes)} hash blocks")

    # Example 4: Batch reconstruction
    print("\n5. Batch reconstruction...")
    input_lengths = [
        len(writer.tokenizer.encode(text)) for text in texts
    ]
    reconstructed_texts = writer.hashes_to_texts(hash_sequences, input_lengths)
    print(f"   Reconstructed {len(reconstructed_texts)} texts")
    for i, (original, reconstructed) in enumerate(
        zip(texts, reconstructed_texts, strict=True)
    ):
        print(f"   Text {i+1}: {original[:30]}... -> {reconstructed[:30]}...")

    print("\n" + "=" * 50)
    print("Example completed!")


if __name__ == "__main__":
    main()
