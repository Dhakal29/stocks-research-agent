"""RAG knowledge store and retriever for investment training books.

Extracts chapters and sections from PDF books in training_books/, chunks them,
and indexes them using TF-IDF and cosine similarity for high-speed local retrieval.
"""

import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)

DEFAULT_BOOKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "training_books")
CACHE_INDEX_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "training_books", ".rag_index.json")
INDEX_VERSION = 2
CHUNK_WORDS = 350
CHUNK_OVERLAP = 50
STOP_WORDS = set("a an and are as at be been by can company for from has have how in is it its of on or per stock that the their this to was were what when which with you your".split())


class BookRAGError(RuntimeError):
    """An actionable local book-index error."""


@dataclass
class DocumentChunk:
    chunk_id: str
    book_title: str
    chapter: str
    page_number: int
    content: str


def _tokenize(text: str) -> list[str]:
    for pattern, replacement in [
        (r"\bp\s*/\s*e\b", "price earnings valuation"),
        (r"\bp\s*/\s*b\b", "price book valuation"),
        (r"\broe\b", "return equity"),
        (r"\broce\b", "return capital employed"),
        (r"\beps\b", "earnings share"),
        (r"\bcfo\b", "cash flow operations"),
    ]:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return [word for word in re.findall(r"\b[^\W\d_]{2,}\b", text.lower()) if word not in STOP_WORDS]


class BookRAGStore:
    def __init__(self, books_dir: str | None = None, index_path: str | None = None) -> None:
        self.books_dir = books_dir or os.getenv("NEPSE_BOOKS_DIR") or DEFAULT_BOOKS_DIR
        self.index_path = index_path or os.getenv("NEPSE_BOOK_INDEX") or str(Path(self.books_dir) / ".rag_index.json")
        self.chunks: list[DocumentChunk] = []
        self.doc_freq: dict[str, int] = {}
        self.tf_idf_vectors: list[dict[str, float]] = []
        self.norms: list[float] = []
        self.book_status: list[dict] = []
        self.warnings: list[str] = []
        self._fingerprint: dict[str, str] = {}

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

    def load_or_build(self, force_rebuild: bool = False, *, ocr: bool = False) -> None:
        """Load existing index if available, otherwise parse PDFs and build index."""
        fingerprint = self._book_fingerprint()
        if not force_rebuild and self.chunks and fingerprint == self._fingerprint and not ocr:
            return
        if not force_rebuild and os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("version") != INDEX_VERSION or data.get("fingerprint") != fingerprint:
                    raise ValueError("Book files or index format have changed")
                if ocr and not data.get("ocr_enabled"):
                    raise ValueError("Rebuilding with OCR enabled")
                self.chunks = [DocumentChunk(**item) for item in data["chunks"]]
                if not self.chunks:
                    raise ValueError("Empty cached index")
                self.book_status = data["books"]
                self.warnings = data["warnings"]
                self._fingerprint = fingerprint
                self._compute_tfidf()
                logger.info("Loaded %d knowledge chunks from %s", len(self.chunks), self.index_path)
                return
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logger.warning("Failed to load cached index, rebuilding: %s", exc)

        self._build_from_pdfs(fingerprint, ocr=ocr)

    def _ocr_pages(self, file_path: Path, digest: str, pages: list[int]) -> dict[str, str]:
        """OCR only pages without usable text; keep source PDFs unchanged."""
        cache_dir = Path(self.index_path).parent / ".ocr_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = cache_dir / f"{digest}.json"
        if cache.exists():
            try:
                data = json.loads(cache.read_text(encoding="utf-8"))
                if all(str(page) in data for page in pages):
                    return data
            except (OSError, ValueError, TypeError):
                pass
        if sys.platform != "darwin" or not shutil.which("swift"):
            raise BookRAGError(
                "The built-in OCR option needs macOS and Swift. On other systems, use OCRmyPDF "
                "to add a text layer to scanned PDFs, then run index-books again."
            )
        script = Path(__file__).with_name("ocr_books.swift")
        # Keep compiler caches in the writable book-cache directory.
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
            raise BookRAGError(f"OCR failed for {file_path.name}. Check the OCR output and retry index-books --ocr.") from exc

    def _build_from_pdfs(self, fingerprint: dict[str, str], *, ocr: bool = False) -> None:
        all_chunks: list[DocumentChunk] = []
        book_status: list[dict] = []
        warnings: list[str] = []

        for fname, digest in fingerprint.items():
            file_path = Path(self.books_dir) / fname
            logger.info("Indexing book: %s", fname)
            try:
                reader = PdfReader(file_path)
                page_texts = [(page.extract_text() or "").strip() for page in reader.pages]
            except Exception as e:
                raise BookRAGError(f"Could not extract {fname}. Check that it is a readable, unencrypted PDF.") from e

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
                    warning += " Run python -m nepse_agent index-books --ocr to include scanned pages."
                warnings.append(warning)
                logger.warning(warning)

            current_chapter = "General / Overview"
            before = len(all_chunks)
            for page_idx, text in enumerate(page_texts):
                if len(text) < 40:
                    continue

                for line in text.split("\n"):
                    clean_line = line.strip()
                    if re.match(r"^chapter\s+\d+", clean_line, re.IGNORECASE):
                        current_chapter = clean_line
                        break

                # Page-bounded, overlapping windows preserve exact PDF page citations.
                words = text.split()
                for p_idx, start in enumerate(range(0, len(words), CHUNK_WORDS - CHUNK_OVERLAP)):
                    para = " ".join(words[start:start + CHUNK_WORDS])
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
                    if start + CHUNK_WORDS >= len(words):
                        break
            book_status.append({
                "filename": fname, "total_pages": len(page_texts),
                "indexed_pages": len(page_texts) - len(skipped), "skipped_pages": skipped,
                "ocr_pages": sorted(int(page) for page, text in ocr_texts.items() if len(text.strip()) >= 40),
                "chunks": len(all_chunks) - before,
            })

        if not all_chunks:
            raise BookRAGError("The books contain no extractable text. Run python -m nepse_agent index-books --ocr for scanned PDFs.")

        self.chunks = all_chunks
        self.book_status = book_status
        self.warnings = warnings
        self._fingerprint = fingerprint
        self._compute_tfidf()

        # Cache index
        temporary_path = None
        try:
            Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=Path(self.index_path).parent, delete=False) as f:
                temporary_path = f.name
                json.dump({
                    "version": INDEX_VERSION, "fingerprint": fingerprint, "ocr_enabled": ocr,
                    "books": book_status, "warnings": warnings,
                    "chunks": [asdict(c) for c in self.chunks],
                }, f, ensure_ascii=False)
            os.replace(temporary_path, self.index_path)
            logger.info("Saved %d chunks to %s", len(self.chunks), self.index_path)
        except OSError as e:
            raise BookRAGError(f"Could not save book index to {self.index_path}. Check its permissions.") from e
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def _compute_tfidf(self) -> None:
        self.doc_freq = {}
        self.tf_idf_vectors = []
        self.norms = []
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
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20.")
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

    def retrieve_for_analysis(self, query: str, evidence: str, top_k: int = 10) -> list[DocumentChunk]:
        """Retrieve several aspects of investing rather than ten similar ratio pages."""
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20.")
        topics = [
            query,
            "due diligence business management corporate governance annual report checklist",
            "earnings profitability return equity revenue profit growth",
            "cash flow operations earnings quality financial statements",
            "debt equity leverage interest coverage financial risk",
            "valuation intrinsic value price earnings price book discounted cash flow",
            "margin safety investment risk dividend sustainability",
        ]
        if re.search(r"\b(bank|banking|insurance|hydropower)\b", evidence, re.IGNORECASE):
            topics.insert(1, "bank banking insurance hydropower sector financial analysis")
        ranked = [self.retrieve(topic, top_k=min(top_k, 5)) for topic in topics]
        # Evidence introduces company-specific concepts; the symbol alone will not
        # occur in educational books, so topical retrieval remains necessary.
        ranked.append(self.retrieve(evidence[:12000], top_k=min(top_k, 5)))
        selected: list[DocumentChunk] = []
        seen_pages: set[tuple[str, int]] = set()
        for rank in range(5):
            for results in ranked:
                if rank >= len(results):
                    continue
                chunk = results[rank]
                page = (chunk.book_title, chunk.page_number)
                if page not in seen_pages:
                    selected.append(chunk)
                    seen_pages.add(page)
                    if len(selected) == top_k:
                        return selected
        return selected

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
    books_dir = os.getenv("NEPSE_BOOKS_DIR") or DEFAULT_BOOKS_DIR
    index_path = os.getenv("NEPSE_BOOK_INDEX") or str(Path(books_dir) / ".rag_index.json")
    if _GLOBAL_STORE is None or _GLOBAL_STORE.books_dir != books_dir or _GLOBAL_STORE.index_path != index_path:
        _GLOBAL_STORE = BookRAGStore()
    _GLOBAL_STORE.load_or_build()
    return _GLOBAL_STORE
