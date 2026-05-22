from __future__ import annotations

from typing import ClassVar

from ollama import ResponseError

from llama_index.embeddings.ollama import OllamaEmbedding


class SafeOllamaEmbedding(OllamaEmbedding):
    SAFE_CHUNK_CHARS: ClassVar[int] = 1800
    SAFE_CHUNK_OVERLAP: ClassVar[int] = 200

    def _get_text_embedding(self, text: str) -> list[float]:
        try:
            return super()._get_text_embedding(text)
        except ResponseError as exc:
            if not self._is_context_length_error(exc):
                raise
            return self._embed_text_safely(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        try:
            return super()._get_text_embeddings(texts)
        except ResponseError as exc:
            if not self._is_context_length_error(exc):
                raise

            print(
                "Encountered overlong embedding input. Falling back to chunked embeddings for this batch.",
                flush=True,
            )
            return [self._embed_text_safely(text) for text in texts]

    def _embed_text_safely(self, text: str) -> list[float]:
        formatted_text = self._format_text(text)
        return self._embed_formatted_text_safely(formatted_text)

    def _embed_formatted_text_safely(self, formatted_text: str) -> list[float]:
        try:
            return self.get_general_text_embedding(formatted_text)
        except ResponseError as exc:
            if not self._is_context_length_error(exc):
                raise

            segments = self._split_for_context(formatted_text)
            if len(segments) == 1:
                return self.get_general_text_embedding(segments[0])

            embeddings = [self._embed_formatted_text_safely(segment) for segment in segments]
            return self._average_embeddings(embeddings)

    def _split_for_context(self, text: str) -> list[str]:
        if len(text) <= self.SAFE_CHUNK_CHARS:
            return [text]

        segments: list[str] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + self.SAFE_CHUNK_CHARS, text_length)
            if end < text_length:
                boundary = text.rfind(" ", start, end)
                if boundary > start + (self.SAFE_CHUNK_CHARS // 2):
                    end = boundary

            segment = text[start:end].strip()
            if segment:
                segments.append(segment)

            if end >= text_length:
                break

            start = max(end - self.SAFE_CHUNK_OVERLAP, start + 1)

        return segments or [text[: self.SAFE_CHUNK_CHARS]]

    @staticmethod
    def _average_embeddings(embeddings: list[list[float]]) -> list[float]:
        if not embeddings:
            raise ValueError("Cannot average an empty embedding list.")

        dimensions = len(embeddings[0])
        totals = [0.0] * dimensions
        for embedding in embeddings:
            for index, value in enumerate(embedding):
                totals[index] += value

        count = float(len(embeddings))
        return [value / count for value in totals]

    @staticmethod
    def _is_context_length_error(exc: ResponseError) -> bool:
        return getattr(exc, "status_code", None) == 400 and "context length" in str(exc).lower()