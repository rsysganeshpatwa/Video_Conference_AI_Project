# utils/text_chunker.py

from typing import List

class TextChunker:
    def __init__(self, tokenizer):
        """
        Initialize with a tokenizer instance.
        """
        self.tokenizer = tokenizer

    def chunk_text(self, text: str, lang: str = "en", max_tokens: int = 400, max_chars: int = 250) -> List[str]:
        """
        Splits the text into chunks respecting both token and character limits.

        Args:
            text (str): Input text to split.
            lang (str): Language code for tokenization.
            max_tokens (int): Maximum number of tokens per chunk.
            max_chars (int): Maximum number of characters per chunk.

        Returns:
            List[str]: A list of text chunks.
        """
        words = text.strip().split()
        chunks = []
        current_chunk = []

        for word in words:
            test_chunk = " ".join(current_chunk + [word])
            token_len = len(self.tokenizer.encode(test_chunk, lang=lang))
            if token_len <= max_tokens and len(test_chunk) <= max_chars:
                current_chunk.append(word)
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
