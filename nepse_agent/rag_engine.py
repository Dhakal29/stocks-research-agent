"""Page-aware semantic chunking and local, persistent vector search for books."""

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np
from pypdf import PdfReader

logger = logging.getLogger(__name__)

DEFAULT_BOOKS_DIR = str(Path(__file__).resolve().parent.parent / "training_books")
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
INDEX_VERSION = 4


class BookRAGError(RuntimeError):
    """An actionable local indexing, embedding or retrieval error."""


@dataclass
class DocumentChunk:
    chunk_id: str
    book_title: str
    chapter: str
    page_number: int
    content: str
    similarity_score: float | None = None


class EmbeddingBackend(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class LocalEmbeddingModel:
    """FastEmbed uses a pretrained ONNX model locally; no provider API key."""

    def __init__(self, model_name: str, cache_dir: str) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding

                logger.info("[RAG embedding model] Loading %s; cache=%s", self.model_name, self.cache_dir)
                self._model = TextEmbedding(
                    model_name=self.model_name, cache_dir=self.cache_dir, threads=2,
                )
            except Exception as exc:
                raise BookRAGError(
                    f"Could not load embedding model {self.model_name}. Install requirements.txt "
                    "and allow the first model download, or configure a cached FastEmbed model."
                ) from exc
        return self._model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(list(self._load().passage_embed(texts, batch_size=32)), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(list(self._load().query_embed(text)), dtype=np.float32)


def _unit_vectors(vectors: np.ndarray, count: int) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != count or not vectors.shape[1] or not np.isfinite(vectors).all():
        raise BookRAGError("The embedding model returned invalid or non-finite vectors.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise BookRAGError("The embedding model returned an empty vector.")
    return vectors / norms


def _sentences(text: str, max_words: int) -> list[str]:
    # Preserve paragraph boundaries and decimal numbers in financial statements.
    parts = re.split(r"(?<=[.!?।])\s+(?=[A-Z0-9\u0900-\u097f])|\n\s*\n", text)
    sentences = []
    for part in parts:
        words = part.split()
        # Unpunctuated tables and unusually long sentences still have a size cap.
        sentences.extend(" ".join(words[start:start + max_words]) for start in range(0, len(words), max_words))
    return sentences


def _informative_passage(text: str) -> bool:
    """Page numbers and isolated headings are not useful investment evidence."""
    words = text.split()
    return len(words) >= 8 and sum(any(char.isalpha() for char in word) for word in words) >= 6


class BookRAGStore:
    def __init__(
        self, books_dir: str | None = None, index_path: str | None = None, *,
        vector_db_path: str | None = None, embedding_model: str | None = None,
        embedder: EmbeddingBackend | None = None,
    ) -> None:
        self.books_dir = str(Path(books_dir or os.getenv("NEPSE_BOOKS_DIR") or DEFAULT_BOOKS_DIR).resolve())
        self.index_path = str(Path(index_path or os.getenv("NEPSE_BOOK_INDEX") or Path(self.books_dir) / ".rag_index.json").resolve())
        self.vector_db_path = str(Path(vector_db_path or os.getenv("NEPSE_VECTOR_DB") or Path(self.index_path).parent / ".rag_vectors").resolve())
        self.embedding_model = embedding_model or os.getenv("NEPSE_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
        self.embedding_cache = str(Path(os.getenv("NEPSE_EMBEDDING_CACHE") or Path(self.index_path).parent / ".embedding_models").resolve())
        try:
            self.min_words = int(os.getenv("NEPSE_CHUNK_MIN_WORDS", "50"))
            self.max_words = int(os.getenv("NEPSE_CHUNK_MAX_WORDS", "180"))
            self.break_percentile = float(os.getenv("NEPSE_SEMANTIC_BREAK_PERCENTILE", "75"))
            self.min_similarity = float(os.getenv("NEPSE_RAG_MIN_SIMILARITY", "0.45"))
            if not 1 <= self.min_words <= self.max_words <= 400 or self.max_words < 8:
                raise ValueError("Chunk sizes must satisfy 1 <= min <= max <= 400, with max at least 8")
            if not 0 < self.break_percentile < 100 or not -1 <= self.min_similarity <= 1:
                raise ValueError("Invalid semantic breakpoint percentile or similarity floor")
        except ValueError as exc:
            raise BookRAGError(f"Invalid RAG configuration: {exc}") from exc
        self._embedder = embedder or LocalEmbeddingModel(self.embedding_model, self.embedding_cache)
        self.chunks: list[DocumentChunk] = []
        self.book_status: list[dict] = []
        self.warnings: list[str] = []
        self._fingerprint: dict[str, str] = {}
        self._client = None
        self._collection = None
        self._chunks_by_id: dict[str, DocumentChunk] = {}
        self._lock = threading.RLock()

    def _index_config(self) -> dict:
        return {
            "embedding_model": self.embedding_model, "metric": "cosine",
            "min_words": self.min_words, "max_words": self.max_words,
            "break_percentile": self.break_percentile,
        }

    def _book_fingerprint(self) -> dict[str, str]:
        directory = Path(self.books_dir)
        if not directory.is_dir():
            raise BookRAGError(f"Books directory does not exist: {directory}")
        paths = sorted(path for path in directory.iterdir() if path.suffix.lower() == ".pdf" and path.is_file())
        if not paths:
            raise BookRAGError(f"No PDF books found in {directory}. Add books before requesting book analysis.")
        try:
            return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        except OSError as exc:
            raise BookRAGError("Could not read the books directory. Check its permissions.") from exc

    def _vector_client(self):
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings

                self._client = chromadb.PersistentClient(
                    path=self.vector_db_path, settings=Settings(anonymized_telemetry=False),
                )
            except Exception as exc:
                raise BookRAGError(
                    f"Could not open Chroma at {self.vector_db_path}. Install requirements.txt "
                    "and check directory permissions."
                ) from exc
        return self._client

    def load_or_build(self, force_rebuild: bool = False, *, ocr: bool = False) -> None:
        """Rebuild legacy indexes; reuse matching semantic chunks and stored vectors."""
        with self._lock:
            fingerprint = self._book_fingerprint()
            if not force_rebuild and self._collection is not None and fingerprint == self._fingerprint and not ocr:
                return
            if Path(self.index_path).exists():
                try:
                    data = json.loads(Path(self.index_path).read_text(encoding="utf-8"))
                    if not isinstance(data, dict):
                        raise ValueError("Invalid index manifest")
                    # Preserve OCR when migrating the old TF-IDF index or changing models.
                    ocr = ocr or bool(data.get("ocr_enabled"))
                    if not force_rebuild and data.get("version") == INDEX_VERSION and data.get("fingerprint") == fingerprint and data.get("config") == self._index_config() and (not ocr or data.get("ocr_enabled")):
                        chunks = [DocumentChunk(**item) for item in data["chunks"]]
                        if not chunks or len({c.chunk_id for c in chunks}) != len(chunks):
                            raise ValueError("Empty or duplicate cached passages")
                        self._index_vectors(chunks, rebuild=False)
                        self.chunks = chunks
                        self._chunks_by_id = {c.chunk_id: c for c in chunks}
                        self.book_status = data["books"]
                        self.warnings = data["warnings"]
                        self._fingerprint = fingerprint
                        logger.info("[RAG index] Loaded %d semantic passages; db=%s", len(chunks), self.vector_db_path)
                        return
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    logger.warning("[RAG index] Cached manifest invalid; rebuilding: %s", exc)
            self._build_from_pdfs(fingerprint, ocr=ocr)

    def _semantic_chunks(self, text: str) -> list[str]:
        sentences = _sentences(text, self.max_words)
        if len(sentences) < 2:
            return [sentence for sentence in sentences if _informative_passage(sentence)]
        embeddings = _unit_vectors(self._embedder.embed_documents(sentences), len(sentences))
        # The page's sentence similarity matrix defines boundaries at meaning shifts.
        similarity_matrix = embeddings @ embeddings.T
        distances = 1.0 - np.diag(similarity_matrix, k=1)
        threshold = float(np.percentile(distances, self.break_percentile))
        groups, current, word_count = [], [], 0
        for index, sentence in enumerate(sentences):
            size = len(sentence.split())
            if current and word_count + size > self.max_words:
                groups.append(" ".join(current))
                current, word_count = [], 0
            current.append(sentence)
            word_count += size
            if index < len(distances) and distances[index] > threshold and word_count >= self.min_words:
                groups.append(" ".join(current))
                current, word_count = [], 0
        if current:
            groups.append(" ".join(current))
        merged = []
        for group in groups:
            if not _informative_passage(group):
                continue
            if merged and len(group.split()) < self.min_words and len(merged[-1].split()) + len(group.split()) <= self.max_words:
                merged[-1] += " " + group
            else:
                merged.append(group)
        return merged

    def _ocr_pages(self, file_path: Path, digest: str, pages: list[int]) -> dict[str, str]:
        cache_dir = Path(self.index_path).parent / ".ocr_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = cache_dir / f"{digest}.json"
        if cache.exists():
            try:
                data = json.loads(cache.read_text(encoding="utf-8"))
                if isinstance(data, dict) and all(isinstance(data.get(str(page)), str) for page in pages):
                    return data
            except (OSError, ValueError, TypeError):
                pass
        if sys.platform != "darwin" or not shutil.which("swift"):
            raise BookRAGError("The built-in OCR option needs macOS and Swift. Otherwise add a text layer with OCRmyPDF.")
        script = Path(__file__).with_name("ocr_books.swift")
        module_cache = cache_dir / "swift_modules"
        module_cache.mkdir(exist_ok=True)
        environment = os.environ.copy()
        environment["CLANG_MODULE_CACHE_PATH"] = str(module_cache.resolve())
        logger.info("Recognizing %d scanned/blank pages in %s", len(pages), file_path.name)
        try:
            subprocess.run(
                ["swift", "-module-cache-path", str(module_cache.resolve()), str(script),
                 str(file_path.resolve()), str(cache.resolve()), ",".join(map(str, pages))],
                check=True, timeout=900, env=environment,
            )
            data = json.loads(cache.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or any(not isinstance(value, str) for value in data.values()):
                raise ValueError("Invalid OCR output")
            return data
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            raise BookRAGError(f"OCR failed for {file_path.name}. Check output and retry load_or_build(ocr=True).") from exc

    def _build_from_pdfs(self, fingerprint: dict[str, str], *, ocr: bool = False) -> None:
        all_chunks, book_status, warnings = [], [], []
        for fname, digest in fingerprint.items():
            file_path = Path(self.books_dir) / fname
            logger.info("[RAG index] Semantic chunking: %s", fname)
            try:
                reader = PdfReader(file_path)
                page_texts = [(page.extract_text() or "").strip() for page in reader.pages]
            except Exception as exc:
                raise BookRAGError(f"Could not extract {fname}. Check that it is a readable, unencrypted PDF.") from exc
            missing = [i + 1 for i, text in enumerate(page_texts) if len(text) < 40]
            ocr_texts = self._ocr_pages(file_path, digest, missing) if ocr and missing else {}
            for page, text in ocr_texts.items():
                index = int(page) - 1
                if index + 1 in missing and len(text.strip()) >= 40:
                    page_texts[index] = text.strip()
            skipped = [i + 1 for i, text in enumerate(page_texts) if len(text) < 40]
            if skipped:
                warning = f"{fname}: {len(skipped)}/{len(page_texts)} pages have no usable text and were skipped."
                if not ocr:
                    warning += " Use BookRAGStore().load_or_build(ocr=True) to include scanned pages."
                warnings.append(warning)
                logger.warning(warning)
            current_chapter = "General / Overview"
            before = len(all_chunks)
            for page_idx, text in enumerate(page_texts):
                if len(text) < 40:
                    continue
                for line in text.splitlines():
                    if re.match(r"^chapter\s+\d+\b", line.strip(), re.IGNORECASE):
                        current_chapter = line.strip()
                        break
                for part, content in enumerate(self._semantic_chunks(text), start=1):
                    all_chunks.append(DocumentChunk(
                        chunk_id=f"{fname}_{page_idx + 1}_{part}", book_title=fname,
                        chapter=current_chapter, page_number=page_idx + 1, content=content,
                    ))
            book_status.append({
                "filename": fname, "total_pages": len(page_texts),
                "indexed_pages": len(page_texts) - len(skipped), "skipped_pages": skipped,
                "ocr_pages": sorted(int(page) for page, text in ocr_texts.items() if len(text.strip()) >= 40),
                "chunks": len(all_chunks) - before,
            })
        if not all_chunks:
            raise BookRAGError("Books have no extractable text. Use load_or_build(ocr=True) for scanned PDFs.")
        self._index_vectors(all_chunks, rebuild=True)
        self._save_manifest({
            "version": INDEX_VERSION, "fingerprint": fingerprint, "config": self._index_config(),
            "ocr_enabled": ocr, "books": book_status, "warnings": warnings,
            "chunks": [asdict(chunk) for chunk in all_chunks],
        })
        self.chunks, self.book_status, self.warnings = all_chunks, book_status, warnings
        self._chunks_by_id = {c.chunk_id: c for c in all_chunks}
        self._fingerprint = fingerprint
        logger.info("[RAG index] Saved %d semantic passages; model=%s metric=cosine db=%s", len(all_chunks), self.embedding_model, self.vector_db_path)

    def _save_manifest(self, data: dict) -> None:
        temporary_path = None
        try:
            Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=Path(self.index_path).parent, delete=False) as file:
                temporary_path = file.name
                json.dump(data, file, ensure_ascii=False)
            os.replace(temporary_path, self.index_path)
        except OSError as exc:
            raise BookRAGError(f"Could not save book index to {self.index_path}. Check permissions.") from exc
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def _index_vectors(self, chunks: list[DocumentChunk], *, rebuild: bool) -> None:
        # A content/config digest keeps old and new embedding spaces separate.
        signature = hashlib.sha256(json.dumps({
            "version": INDEX_VERSION, "config": self._index_config(),
            "chunks": [asdict(chunk) for chunk in chunks],
        }, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        try:
            collection = self._vector_client().get_or_create_collection(
                name=f"books_{signature[:32]}", embedding_function=None,
                configuration={"hnsw": {"space": "cosine"}},
                metadata={"embedding_model": self.embedding_model, "signature": signature},
            )
            if rebuild or collection.count() != len(chunks):
                logger.info("[RAG index] Embedding %d passages into %s", len(chunks), collection.name)
                for start in range(0, len(chunks), 64):
                    batch = chunks[start:start + 64]
                    vectors = _unit_vectors(self._embedder.embed_documents([c.content for c in batch]), len(batch))
                    collection.upsert(
                        ids=[c.chunk_id for c in batch], embeddings=vectors.tolist(),
                        documents=[c.content for c in batch],
                        metadatas=[{"book_title": c.book_title, "chapter": c.chapter, "page_number": c.page_number} for c in batch],
                    )
            if collection.count() != len(chunks):
                raise BookRAGError("The persisted vector count does not match the book passages. Rebuild the index.")
            self._collection = collection
        except BookRAGError:
            raise
        except Exception as exc:
            raise BookRAGError("Could not create or update the Chroma book collection. Check vector-store permissions and rebuild.") from exc

    def retrieve(self, query: str, top_k: int = 5) -> list[DocumentChunk]:
        """Embed the exact logged query and rank passages by cosine similarity."""
        if not isinstance(top_k, int) or not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20.")
        if not isinstance(query, str):
            raise ValueError("The vector query must be text.")
        query = query.strip()
        logger.info("[RAG vector query] text=%r top_k=%d min_similarity=%.3f model=%s", query, top_k, self.min_similarity, self.embedding_model)
        with self._lock:
            if not query or not self.chunks or self._collection is None:
                logger.info("[RAG vector results] No query text or loaded book collection.")
                return []
            try:
                vector = _unit_vectors(self._embedder.embed_query(query), 1)
                result = self._collection.query(
                    query_embeddings=vector.tolist(), n_results=min(top_k * 3, len(self.chunks)),
                    include=["distances"],
                )
                selected = []
                for chunk_id, distance in zip(result["ids"][0], result["distances"][0]):
                    score = float(np.clip(1.0 - distance, -1.0, 1.0))
                    if score < self.min_similarity:
                        logger.info("[RAG vector result rejected] id=%s cosine_similarity=%.4f", chunk_id, score)
                        continue
                    chunk = replace(self._chunks_by_id[chunk_id], similarity_score=score)
                    if not _informative_passage(chunk.content):
                        logger.info("[RAG vector result rejected] id=%s reason=short_or_nontext", chunk_id)
                        continue
                    selected.append(chunk)
                    logger.info(
                        "[RAG vector result] id=%s cosine_similarity=%.4f book=%s chapter=%s page=%d\n%s",
                        chunk.chunk_id, score, chunk.book_title, chunk.chapter, chunk.page_number, chunk.content,
                    )
                    if len(selected) == top_k:
                        break
                logger.info("[RAG vector results] Selected %d passages from %s", len(selected), self._collection.name)
                return selected
            except BookRAGError:
                raise
            except Exception as exc:
                raise BookRAGError("Semantic vector search failed. Check the embedding model and rebuild the book index.") from exc

    def retrieve_for_analysis(self, query: str, evidence: str = "", top_k: int = 8) -> list[DocumentChunk]:
        """Prioritize investment methods, then search meaningful evidence sections."""
        if not isinstance(top_k, int) or not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20.")
        # A ticker alone has no investment meaning and can match page numbers.
        topics = [] if re.fullmatch(r"[A-Za-z][A-Za-z0-9]{1,11}", query.strip()) else [query]
        topics.extend([
            "Assess business quality, competitive advantage and management using an investment due diligence checklist.",
            "Evaluate sustainable earnings growth, profitability and return on equity.",
            "Check whether operating cash flow supports accounting profit and dividends.",
            "Assess financial leverage, debt repayment and interest coverage.",
            "Compare price to intrinsic value, growth and peers; assess valuation and margin of safety.",
        ])
        if re.search(r"\b(bank|banking|insurance|hydro\s*power)\b", query + " " + evidence, re.IGNORECASE):
            topics.append("Sector-specific analysis of banks, insurance or hydropower: relevant profitability and financial risk measures.")
        if evidence:
            logger.info("[RAG evidence input] Web response used to construct vector queries:\n%s", evidence)
            # Separate short sections avoid silently sending an entire report into
            # an embedding model's limited context window.
            body = evidence.split("## Cited sources", 1)[0]
            body = re.sub(r"\[[^\]\n]*\]\(https?://[^\s)]*\)", "", body)
            body = re.sub(r"https?://\S+", "", body)
            body = re.sub(r"(?m)^Researched at:.*$", "", body)
            sections = re.split(r"\n#{1,6}\s|\n\s*\n", body)
            evidence_topics = []
            for section in sections:
                words = section.split()
                if len(words) >= 8:
                    for start in range(0, len(words), self.max_words):
                        evidence_topics.append(" ".join(words[start:start + self.max_words]))
            topics.extend(evidence_topics[:8])
        ranked = [self.retrieve(topic, top_k=min(top_k, 4)) for topic in dict.fromkeys(topics) if topic.strip()]
        selected, seen = [], set()
        for rank in range(4):
            for results in ranked:
                if rank < len(results) and results[rank].chunk_id not in seen:
                    selected.append(results[rank])
                    seen.add(results[rank].chunk_id)
                    if len(selected) == top_k:
                        return selected
        return selected

    def get_core_due_diligence_framework(self) -> str:
        chunks = self.retrieve_for_analysis("Investment due diligence and valuation principles", top_k=6)
        return "\n\n".join(f"[{c.book_title} | {c.chapter} | PDF page {c.page_number}]:\n{c.content}" for c in chunks)


_GLOBAL_STORE: BookRAGStore | None = None
_STORE_LOCK = threading.Lock()


def get_rag_store() -> BookRAGStore:
    global _GLOBAL_STORE
    candidate = BookRAGStore()
    with _STORE_LOCK:
        if _GLOBAL_STORE is None or (
            _GLOBAL_STORE.books_dir, _GLOBAL_STORE.index_path, _GLOBAL_STORE.vector_db_path,
            _GLOBAL_STORE.embedding_cache, _GLOBAL_STORE._index_config(),
        ) != (
            candidate.books_dir, candidate.index_path, candidate.vector_db_path,
            candidate.embedding_cache, candidate._index_config(),
        ):
            _GLOBAL_STORE = candidate
        _GLOBAL_STORE.min_similarity = candidate.min_similarity
        _GLOBAL_STORE.load_or_build()
        return _GLOBAL_STORE
