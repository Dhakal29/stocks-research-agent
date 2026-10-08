"""RAG knowledge store and retriever for investment training books.

Extracts chapters and sections from PDF books in training_books/, chunks them,
and indexes them using TF-IDF and cosine similarity for high-speed local retrieval.
"""

import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

from pypdf import PdfReader

logger = logging.getLogger(__name__)

DEFAULT_BOOKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "training_books")
CACHE_INDEX_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "training_books", ".rag_index.json")


@dataclass
class DocumentChunk:
    chunk_id: str
    book_title: str
    chapter: str
    page_number: int
    content: str


def _tokenize(text: str) -> list[str]:
    return [word.lower() for word in re.findall(r"\b[A-Za-z0-9_]{2,}\b", text)]


class BookRAGStore:
    def __init__(self, books_dir: str = DEFAULT_BOOKS_DIR, index_path: str = CACHE_INDEX_FILE) -> None:
        self.books_dir = books_dir
        self.index_path = index_path
        self.chunks: list[DocumentChunk] = []
        self.doc_freq: dict[str, int] = {}
        self.tf_idf_vectors: list[dict[str, float]] = []
        self.norms: list[float] = []

    def load_or_build(self, force_rebuild: bool = False) -> None:
        """Load existing index if available, otherwise parse PDFs and build index."""
        if not force_rebuild and os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.chunks = [DocumentChunk(**item) for item in data["chunks"]]
                self._compute_tfidf()
                logger.info("Loaded %d knowledge chunks from %s", len(self.chunks), self.index_path)
                return
            except Exception as exc:
                logger.warning("Failed to load cached index, rebuilding: %s", exc)

        self._build_from_pdfs()

    def _build_from_pdfs(self) -> None:
        if not os.path.exists(self.books_dir):
            logger.warning("Books directory does not exist: %s", self.books_dir)
            return

        pdf_files = [f for f in os.listdir(self.books_dir) if f.endswith(".pdf")]
        all_chunks: list[DocumentChunk] = []

        for fname in sorted(pdf_files):
            file_path = os.path.join(self.books_dir, fname)
            logger.info("Indexing book: %s", fname)
            try:
                reader = PdfReader(file_path)
            except Exception as e:
                logger.error("Error opening %s: %s", fname, e)
                continue

            current_chapter = "General / Overview"
            for page_idx, page in enumerate(reader.pages):
                text = (page.extract_text() or "").strip()
                if not text:
                    continue

                for line in text.split("\n"):
                    clean_line = line.strip()
                    if re.match(r"^chapter\s+\d+", clean_line, re.IGNORECASE) or re.match(r"^module\s+\d+", clean_line, re.IGNORECASE):
                        current_chapter = clean_line
                        break

                paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 60]
                if not paragraphs:
                    paragraphs = [text]

                for p_idx, para in enumerate(paragraphs):
                    chunk_id = f"{fname}_{page_idx+1}_{p_idx+1}"
                    all_chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            book_title=fname,
                            chapter=current_chapter,
                            page_number=page_idx + 1,
                            content=para,
                        )
                    )

        self.chunks = all_chunks
        self._compute_tfidf()

        # Cache index
        try:
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump({"chunks": [asdict(c) for c in self.chunks]}, f)
            logger.info("Saved %d chunks to %s", len(self.chunks), self.index_path)
        except Exception as e:
            logger.warning("Could not cache index to file: %s", e)

    def _compute_tfidf(self) -> None:
        total_docs = len(self.chunks)
        if total_docs == 0:
            return

        self.doc_freq = {}
        doc_tokens_list: list[list[str]] = []

        for chunk in self.chunks:
            tokens = _tokenize(chunk.content) + _tokenize(chunk.chapter) * 2
            doc_tokens_list.append(tokens)
            unique_tokens = set(tokens)
            for t in unique_tokens:
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1

        self.tf_idf_vectors = []
        self.norms = []

        for tokens in doc_tokens_list:
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1

            vec: dict[str, float] = {}
            norm_sq = 0.0
            for t, count in tf.items():
                idf = math.log((total_docs + 1) / (self.doc_freq[t] + 1)) + 1.0
                score = (1 + math.log(count)) * idf
                vec[t] = score
                norm_sq += score * score

            self.tf_idf_vectors.append(vec)
            self.norms.append(math.sqrt(norm_sq) if norm_sq > 0 else 1.0)

    def retrieve(self, query: str, top_k: int = 5) -> list[DocumentChunk]:
        """Retrieve most relevant chunks from the books for a given query."""
        if not self.chunks or not self.tf_idf_vectors:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        total_docs = len(self.chunks)
        q_tf: dict[str, int] = {}
        for t in query_tokens:
            q_tf[t] = q_tf.get(t, 0) + 1

        q_vec: dict[str, float] = {}
        q_norm_sq = 0.0
        for t, count in q_tf.items():
            if t in self.doc_freq:
                idf = math.log((total_docs + 1) / (self.doc_freq[t] + 1)) + 1.0
                score = (1 + math.log(count)) * idf
                q_vec[t] = score
                q_norm_sq += score * score

        q_norm = math.sqrt(q_norm_sq) if q_norm_sq > 0 else 1.0

        scores: list[tuple[float, int]] = []
        for idx, vec in enumerate(self.tf_idf_vectors):
            dot_product = sum(score * vec.get(term, 0.0) for term, score in q_vec.items())
            sim = dot_product / (q_norm * self.norms[idx])
            if sim > 0:
                scores.append((sim, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [self.chunks[idx] for _, idx in scores[:top_k]]

    def get_core_due_diligence_framework(self) -> str:
        """Always retrieve the canonical equity due diligence & valuation rules from the books."""
        relevant_chunks = self.retrieve(
            "due diligence checklist return on equity gross profit margin debt level cash flow from operations valuation DCF intrinsic value Graham",
            top_k=6,
        )
        snippets = []
        for chunk in relevant_chunks:
            snippets.append(
                f"[{chunk.book_title} | {chunk.chapter} | Page {chunk.page_number}]:\n{chunk.content}"
            )
        return "\n\n".join(snippets)


# Global singleton instance
_GLOBAL_STORE: BookRAGStore | None = None


def get_rag_store() -> BookRAGStore:
    global _GLOBAL_STORE
    if _GLOBAL_STORE is None:
        _GLOBAL_STORE = BookRAGStore()
        _GLOBAL_STORE.load_or_build()
    return _GLOBAL_STORE
