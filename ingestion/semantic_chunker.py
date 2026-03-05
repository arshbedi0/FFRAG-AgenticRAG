"""
semantic_chunker.py
────────────────────
Replaces the fixed 800-char chunker in ingest_to_chroma.py.

Strategy: Sentence-Window Chunking
  - Split document into individual sentences
  - Embed each sentence (small, precise vector)
  - At retrieval time, inject ±window_size surrounding sentences into LLM prompt
  - Result: precise retrieval + rich context for generation

Parent-Child also supported:
  - Child chunks (2-3 sentences) are embedded
  - If ≥3 children from same parent are retrieved, swap for full parent paragraph

Usage:
  from ingestion.semantic_chunker import SemanticChunker
  chunker = SemanticChunker()
  chunks  = chunker.chunk_pdf(pdf_path)
  # Each chunk: {"text": ..., "window": ..., "parent": ..., "metadata": ...}
"""


import re
import os
from pathlib import Path

# Universal config loader for local (.env) and Streamlit Cloud (st.secrets)
def get_config(var, default=None, cast_type=None):
    try:
        import streamlit as st
        value = st.secrets.get(var, None)
        if value is not None:
            return cast_type(value) if cast_type else value
    except ImportError:
        pass
    value = os.getenv(var, default)
    return cast_type(value) if cast_type and value is not None else value


class SemanticChunker:
    """
    Sentence-window chunker with parent-child support.
    No ML model needed — pure rule-based sentence splitting.
    """

    def __init__(
        self,
        window_size: int = 3,       # sentences either side of target
        child_size:  int = 2,       # sentences per child chunk
        parent_size: int = 8,       # sentences per parent block
        min_sentence_len: int = 20, # ignore very short fragments
    ):
        self.window_size      = window_size
        self.child_size       = child_size
        self.parent_size      = parent_size
        self.min_sentence_len = min_sentence_len

    # ── Sentence splitter ──────────────────────────────────────
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into clean sentences."""
        # Normalise whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Split on sentence-ending punctuation followed by space + capital
        # Handles: "Mr.", "Fig.", numbered lists, abbreviations
        raw = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9\"])', text)

        sentences = []
        for s in raw:
            s = s.strip()
            if len(s) >= self.min_sentence_len:
                sentences.append(s)

        return sentences

    # ── Extract text from PDF ──────────────────────────────────
    def _extract_text(self, pdf_path: str) -> str:
        """Extract full text from PDF using pdfplumber."""
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pip install pdfplumber")

        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n\n".join(pages)

    # ── Build sentence windows ─────────────────────────────────
    def _build_windows(
        self, sentences: list[str], filename: str, doc_name: str
    ) -> list[dict]:
        """
        For each sentence, build:
          - text   : the sentence itself (what gets embedded)
          - window : ±window_size surrounding sentences (what LLM reads)
          - parent : the larger paragraph block this sentence belongs to
        """
        chunks = []
        n = len(sentences)

        for i, sentence in enumerate(sentences):
            # Window: surrounding context
            start  = max(0, i - self.window_size)
            end    = min(n, i + self.window_size + 1)
            window = " ".join(sentences[start:end])

            # Parent block: larger paragraph context
            p_start = (i // self.parent_size) * self.parent_size
            p_end   = min(n, p_start + self.parent_size)
            parent  = " ".join(sentences[p_start:p_end])

            # Parent ID for auto-merging
            parent_id = f"{filename}_parent_{p_start}"

            chunks.append({
                "text":      sentence,      # embedded
                "window":    window,        # injected into LLM prompt
                "parent":    parent,        # used for auto-merging
                "parent_id": parent_id,
                "metadata": {
                    "filename":       filename,
                    "doc_name":       doc_name,
                    "sentence_idx":   i,
                    "n_sentences":    n,
                    "window_start":   start,
                    "window_end":     end - 1,
                    "parent_id":      parent_id,
                    "chunk_strategy": "sentence_window",
                }
            })

        return chunks

    # ── Build child chunks ─────────────────────────────────────
    def _build_children(
        self, sentences: list[str], filename: str
    ) -> list[dict]:
        """
        Group sentences into child chunks of child_size.
        Used for parent-child (auto-merging) retrieval.
        """
        children = []
        for i in range(0, len(sentences), self.child_size):
            group     = sentences[i:i + self.child_size]
            child_txt = " ".join(group)

            # Parent: the parent_size block this child belongs to
            p_start   = (i // self.parent_size) * self.parent_size
            p_end     = min(len(sentences), p_start + self.parent_size)
            parent    = " ".join(sentences[p_start:p_end])
            parent_id = f"{filename}_parent_{p_start}"

            children.append({
                "text":      child_txt,
                "parent":    parent,
                "parent_id": parent_id,
                "metadata": {
                    "filename":       filename,
                    "child_start":    i,
                    "child_end":      i + len(group) - 1,
                    "parent_id":      parent_id,
                    "chunk_strategy": "parent_child",
                }
            })

        return children

    # ── Public API ─────────────────────────────────────────────
    def chunk_pdf(
        self, pdf_path: str, strategy: str = "sentence_window"
    ) -> list[dict]:
        """
        Chunk a PDF using sentence-window or parent-child strategy.

        Args:
            pdf_path : path to PDF file
            strategy : "sentence_window" | "parent_child" | "both"

        Returns:
            List of chunk dicts with text, window/parent, metadata
        """
        filename = Path(pdf_path).name
        doc_name = filename.replace(".pdf", "").replace("-", " ").replace("_", " ")

        print(f"  Chunking: {filename}")
        text      = self._extract_text(pdf_path)
        sentences = self._split_sentences(text)
        print(f"    Extracted {len(sentences)} sentences")

        chunks = []

        if strategy in ("sentence_window", "both"):
            windows = self._build_windows(sentences, filename, doc_name)
            chunks.extend(windows)
            print(f"    Built {len(windows)} sentence-window chunks")

        if strategy in ("parent_child", "both"):
            children = self._build_children(sentences, filename)
            chunks.extend(children)
            print(f"    Built {len(children)} parent-child chunks")

        return chunks

    def chunk_text(
        self, text: str, filename: str, doc_name: str,
        strategy: str = "sentence_window"
    ) -> list[dict]:
        """Chunk raw text (for non-PDF sources)."""
        sentences = self._split_sentences(text)
        chunks    = []
        if strategy in ("sentence_window", "both"):
            chunks.extend(self._build_windows(sentences, filename, doc_name))
        if strategy in ("parent_child", "both"):
            chunks.extend(self._build_children(sentences, filename))
        return chunks


# ══════════════════════════════════════════════════════════════
# AUTO-MERGE RETRIEVAL HELPER
# ══════════════════════════════════════════════════════════════
class AutoMerger:
    """
    After retrieval, check if ≥ merge_threshold child chunks
    from the same parent were retrieved. If so, swap them all
    for the single parent block — richer context, fewer tokens.
    """

    def __init__(self, merge_threshold: int = 3):
        self.merge_threshold = merge_threshold

    def merge(self, retrieved_chunks: list[dict]) -> list[dict]:
        """
        Input:  list of retrieved child chunks
        Output: list with parent blocks substituted where applicable
        """
        from collections import Counter

        # Count how many children per parent were retrieved
        parent_counts = Counter(
            c["metadata"].get("parent_id") for c in retrieved_chunks
            if c["metadata"].get("chunk_strategy") == "parent_child"
        )

        merged    = []
        used_parents = set()

        for chunk in retrieved_chunks:
            pid = chunk["metadata"].get("parent_id")

            # If this parent qualifies for merging
            if (pid and parent_counts[pid] >= self.merge_threshold
                    and pid not in used_parents):
                # Substitute parent block
                merged.append({
                    "document": chunk.get("parent", chunk["document"]),
                    "metadata": {
                        **chunk["metadata"],
                        "merged":          True,
                        "children_merged": parent_counts[pid],
                        "chunk_strategy":  "auto_merged",
                    }
                })
                used_parents.add(pid)

            elif pid not in used_parents:
                # Use window text instead of bare sentence for LLM
                merged.append({
                    "document": chunk.get("window", chunk["document"]),
                    "metadata": chunk["metadata"],
                })

        return merged


# ══════════════════════════════════════════════════════════════
# RE-INGESTION HELPER
# ══════════════════════════════════════════════════════════════
def reingest_regulations_semantic(
    regulations_dir: str = "data/regulations",
    chroma_dir:      str = "chroma_db",
    collection_name: str = "regulations_v2",
    strategy:        str = "sentence_window",
):
    """
    Drop-in replacement for the regulations ingestion in ingest_to_chroma.py.
    Creates a new collection 'regulations_v2' with semantic chunks.

    Run standalone:
      python ingestion/semantic_chunker.py
    """
    import chromadb
    from chromadb.utils import embedding_functions
    from pathlib import Path
    import os
    from dotenv import load_dotenv
    load_dotenv()

    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    client = chromadb.PersistentClient(path=chroma_dir)

    # Drop old v2 collection if exists
    try:
        client.delete_collection(collection_name)
        print(f"Dropped existing {collection_name}")
    except Exception:
        pass

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.create_collection(collection_name, embedding_function=ef)
    chunker    = SemanticChunker()

    pdf_files = list(Path(regulations_dir).glob("*.pdf"))
    print(f"\nFound {len(pdf_files)} PDFs in {regulations_dir}")

    all_chunks = []
    for pdf_path in pdf_files:
        chunks = chunker.chunk_pdf(str(pdf_path), strategy=strategy)
        all_chunks.extend(chunks)

    print(f"\nTotal chunks: {len(all_chunks)} — ingesting into ChromaDB...")

    # Batch ingest
    BATCH = 100
    for i in range(0, len(all_chunks), BATCH):
        batch = all_chunks[i:i + BATCH]
        collection.add(
            ids        = [f"reg_v2_{i+j}" for j in range(len(batch))],
            documents  = [c["text"] for c in batch],      # sentence embedded
            metadatas  = [{
                **c["metadata"],
                "window": c.get("window", ""),             # stored for retrieval
                "parent": c.get("parent", ""),
            } for c in batch],
        )
        print(f"  Ingested {min(i+BATCH, len(all_chunks))}/{len(all_chunks)}")

    print(f"\n✅ regulations_v2 collection: {collection.count()} documents")
    return collection


if __name__ == "__main__":
    reingest_regulations_semantic()
