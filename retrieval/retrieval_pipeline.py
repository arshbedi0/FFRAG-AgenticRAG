"""
retrieval_pipeline.py
──────────────────────
Hybrid retrieval: BM25 + Dense (ChromaDB) + Reranker with RRF fusion.

Queries all three collections:
  - transactions   : structured transaction records
  - graph_captions : LLaVA wallet network descriptions
  - regulations    : FATF / FinCEN / OCC regulatory chunks

Run standalone to test:
  python retrieval/retrieval_pipeline.py

Import in your app:
  from retrieval.retrieval_pipeline import ForensicsRetriever
  retriever = ForensicsRetriever()
  results = retriever.retrieve("show me structuring patterns")
"""

import os, json
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ──
CHROMA_DIR      = os.getenv("CHROMA_DIR",       "chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL",  "BAAI/bge-small-en-v1.5")
RERANKER_MODEL  = os.getenv("RERANKER_MODEL",   "cross-encoder/ms-marco-MiniLM-L-6-v2")
TOP_K_EACH      = int(os.getenv("TOP_K_EACH",   "10"))   # per retriever per collection
TOP_K_RERANK    = int(os.getenv("TOP_K_RERANK", "5"))    # final chunks sent to LLM

# ── IMPORTS ──
try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    raise ImportError("pip install chromadb sentence-transformers")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("pip install rank-bm25")

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    raise ImportError("pip install sentence-transformers")


# ══════════════════════════════════════════════════════════════
# BM25 INDEX — built in memory from Chroma documents
# ══════════════════════════════════════════════════════════════
class BM25Index:
    """
    Builds a BM25 index from all documents in a Chroma collection.
    Rebuilt each time (fast enough for our dataset size).
    """
    def __init__(self, collection):
        self.collection = collection
        self._build()

    def _build(self):
        # Pull all documents from Chroma
        results = self.collection.get(include=["documents", "metadatas"])
        self.all_ids       = results["ids"]
        self.all_docs      = results["documents"]
        self.all_metadatas = results["metadatas"]

        # Tokenise for BM25 (simple whitespace split, lowercase)
        tokenised = [doc.lower().split() for doc in self.all_docs]
        self.bm25 = BM25Okapi(tokenised)

    def search(self, query: str, top_k: int) -> list[dict]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        # Get top_k indices sorted by score descending
        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        return [
            {
                "id":       self.all_ids[i],
                "document": self.all_docs[i],
                "metadata": self.all_metadatas[i],
                "score":    float(scores[i]),
                "method":   "bm25",
            }
            for i in ranked if scores[i] > 0  # filter zero-score docs
        ]


# ══════════════════════════════════════════════════════════════
# RECIPROCAL RANK FUSION
# ══════════════════════════════════════════════════════════════
def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60
) -> list[dict]:
    """
    Merge multiple ranked lists into one using RRF.
    k=60 is the standard constant that balances high/low rank influence.

    Score formula: sum(1 / (k + rank)) across all lists where doc appears.
    Documents appearing high in multiple lists get the highest scores.
    """
    scores  = {}   # doc_id → rrf_score
    doc_map = {}   # doc_id → full doc dict

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            doc_id = doc["id"]
            scores[doc_id]  = scores.get(doc_id, 0.0) + (1.0 / (k + rank))
            doc_map[doc_id] = doc   # keep latest copy (same doc)

    # Sort by RRF score descending
    fused = sorted(doc_map.values(), key=lambda d: scores[d["id"]], reverse=True)

    # Attach RRF score to each doc
    for doc in fused:
        doc["rrf_score"] = scores[doc["id"]]

    return fused


# ══════════════════════════════════════════════════════════════
# MAIN RETRIEVER CLASS
# ══════════════════════════════════════════════════════════════
class ForensicsRetriever:

    def __init__(self):
        print("🔧 Initialising ForensicsRetriever...")

        # Chroma client + collections
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self.col_txn  = self.client.get_collection("transactions",    embedding_function=self.ef)
        self.col_cap  = self.client.get_collection("graph_captions",  embedding_function=self.ef)
        self.col_reg  = self.client.get_collection("regulations",     embedding_function=self.ef)

        # BM25 indexes (one per collection)
        print("   Building BM25 indexes...")
        self.bm25_txn  = BM25Index(self.col_txn)
        self.bm25_cap  = BM25Index(self.col_cap)
        self.bm25_reg  = BM25Index(self.col_reg)

        # Cross-encoder reranker
        print(f"   Loading reranker: {RERANKER_MODEL}")
        self.reranker = CrossEncoder(RERANKER_MODEL, max_length=512)

        print("✅ ForensicsRetriever ready.\n")

    # ──────────────────────────────────────────────
    # DENSE RETRIEVAL (Chroma vector search)
    # ──────────────────────────────────────────────
    def _dense_search(self, collection, query: str, top_k: int) -> list[dict]:
        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"]
        )
        docs = []
        for i in range(len(results["ids"][0])):
            docs.append({
                "id":       results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score":    1 - results["distances"][0][i],  # convert distance → similarity
                "method":   "dense",
            })
        return docs

    # ──────────────────────────────────────────────
    # RERANKER
    # ──────────────────────────────────────────────
    def _rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return []

        # Build (query, document) pairs for cross-encoder
        pairs = [(query, doc["document"][:500]) for doc in candidates]  # truncate for speed
        scores = self.reranker.predict(pairs)

        # Attach reranker score and sort
        for doc, score in zip(candidates, scores):
            doc["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda d: d["rerank_score"], reverse=True)
        return reranked[:top_k]

    # ──────────────────────────────────────────────
    # QUERY REWRITER
    # Expands a vague user query into better search terms
    # ──────────────────────────────────────────────
    def _rewrite_query(self, query: str) -> list[str]:
        """
        Simple rule-based query expansion.
        In the full app this is replaced by an LLM rewriter.
        """
        queries = [query]  # always include original

        q = query.lower()

        # AML typology expansion
        if any(w in q for w in ["structur", "threshold", "10000", "9999"]):
            queries.append("structuring transactions below CTR reporting threshold £10000")
            queries.append("SAR filing requirements structuring BSA")

        if any(w in q for w in ["smurf", "aggregat", "hub", "funnel"]):
            # FATF does not use the word "smurfing" — their terminology is
            # placement, aggregation, and layering. Expanding to both ensures
            # we hit regulatory docs AND transaction records.
            queries.append("smurfing multiple senders aggregating into one account")
            queries.append("placement layering integration money laundering")
            queries.append("placement stage aggregation multiple deposits single account")
            queries.append("FATF placement layering integration three stages money laundering")
            queries.append("structuring aggregation typology multiple small transactions")

        if any(w in q for w in ["layer", "chain", "hop", "transit"]):
            queries.append("layering sequential transfers obscure audit trail")
            queries.append("ACH transfers rapid movement multiple accounts")

        if any(w in q for w in ["corridor", "high risk", "uae", "turkey", "morocco"]):
            queries.append("high risk geographic corridor cross border suspicious")
            queries.append("FATF high risk jurisdictions correspondent banking")

        if any(w in q for w in ["dormant", "inactive", "reactivat"]):
            queries.append("dormant account reactivation large transaction placement")

        if any(w in q for w in ["currency", "mismatch", "forex"]):
            queries.append("currency mismatch received currency inconsistent location")
            queries.append("trade based money laundering TBML invoice")

        if any(w in q for w in ["sar", "suspicious activity report", "filing"]):
            queries.append("SAR suspicious activity report filing requirements FinCEN")
            queries.append("BSA bank secrecy act reporting obligations")

        if any(w in q for w in ["fatf", "recommendation", "regulation", "compliance"]):
            queries.append("FATF recommendations AML CFT compliance obligations")

        return list(dict.fromkeys(queries))  # deduplicate while preserving order

    # ──────────────────────────────────────────────
    # MAIN RETRIEVE METHOD
    # ──────────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RERANK,
        collections: Optional[list[str]] = None,
        verbose: bool = False,
    ) -> dict:
        """
        Full hybrid retrieval pipeline.

        Args:
            query:       User's natural language question
            top_k:       Number of final chunks to return (after reranking)
            collections: Which collections to search. None = all three.
                         Options: ["transactions", "graph_captions", "regulations"]
            verbose:     Print intermediate results

        Returns:
            {
                "query":        original query,
                "rewritten":    expanded queries used,
                "transactions": [...],   top chunks from transactions
                "graph_captions": [...], top chunks from captions
                "regulations":  [...],   top chunks from regulations
                "all_results":  [...],   all chunks merged + reranked
            }
        """
        if collections is None:
            collections = ["transactions", "graph_captions", "regulations"]

        # Step 1: Query rewriting
        rewritten_queries = self._rewrite_query(query)
        if verbose:
            print(f"\n📝 Query rewritten into {len(rewritten_queries)} variants:")
            for q in rewritten_queries:
                print(f"   • {q}")

        # Step 2: Retrieve from each collection using both BM25 + dense
        # We aggregate across all rewritten query variants
        all_candidates = []
        collection_results = {c: [] for c in collections}

        col_map = {
            "transactions":   (self.col_txn,  self.bm25_txn),
            "graph_captions": (self.col_cap,  self.bm25_cap),
            "regulations":    (self.col_reg,  self.bm25_reg),
        }

        for col_name in collections:
            chroma_col, bm25_idx = col_map[col_name]
            col_candidates = []

            for q in rewritten_queries[:3]:  # limit to top 3 rewrites for speed
                # Dense retrieval
                dense_results = self._dense_search(chroma_col, q, TOP_K_EACH)

                # BM25 retrieval
                bm25_results  = bm25_idx.search(q, TOP_K_EACH)

                # RRF fusion for this query
                fused = reciprocal_rank_fusion([dense_results, bm25_results])
                col_candidates.extend(fused)

            # Deduplicate by ID (keep highest rrf_score)
            seen = {}
            for doc in col_candidates:
                doc_id = doc["id"]
                if doc_id not in seen or doc.get("rrf_score", 0) > seen[doc_id].get("rrf_score", 0):
                    seen[doc_id] = doc
            col_candidates = list(seen.values())

            # Tag with collection name
            for doc in col_candidates:
                doc["collection"] = col_name

            collection_results[col_name] = col_candidates
            all_candidates.extend(col_candidates)

            if verbose:
                print(f"\n   [{col_name}] {len(col_candidates)} candidates after fusion")

        # Step 3: Rerank ALL candidates together
        if verbose:
            print(f"\n🔍 Reranking {len(all_candidates)} total candidates → top {top_k}...")

        final_results = self._rerank(query, all_candidates, top_k)

        # Step 4: Separate back by collection for structured output
        output = {
            "query":          query,
            "rewritten":      rewritten_queries,
            "all_results":    final_results,
            "transactions":   [r for r in final_results if r.get("collection") == "transactions"],
            "graph_captions": [r for r in final_results if r.get("collection") == "graph_captions"],
            "regulations":    [r for r in final_results if r.get("collection") == "regulations"],
        }

        return output

    # ──────────────────────────────────────────────
    # FORMAT FOR LLM CONTEXT
    # ──────────────────────────────────────────────
    def format_context(self, results: dict) -> str:
        """
        Formats retrieval results into a clean context block for the LLM prompt.
        """
        sections = []

        if results["transactions"]:
            sections.append("=== TRANSACTION RECORDS ===")
            for i, r in enumerate(results["transactions"], 1):
                meta = r["metadata"]
                sections.append(
                    f"[TXN-{i}] {r['document']}\n"
                    f"   → Typology: {meta.get('typology','?')} | "
                    f"Suspicious: {'YES' if meta.get('is_suspicious') else 'NO'} | "
                    f"Amount: £{meta.get('amount', 0):,.2f} | "
                    f"Rerank score: {r.get('rerank_score', 0):.3f}"
                )

        if results["graph_captions"]:
            sections.append("\n=== WALLET NETWORK GRAPH ANALYSIS ===")
            for i, r in enumerate(results["graph_captions"], 1):
                sections.append(
                    f"[GRAPH-{i}] {r['document'][:600]}...\n"
                    f"   → Graph: {r['metadata'].get('graph_id','?')} | "
                    f"Rerank score: {r.get('rerank_score', 0):.3f}"
                )

        if results["regulations"]:
            sections.append("\n=== REGULATORY REFERENCES ===")
            for i, r in enumerate(results["regulations"], 1):
                sections.append(
                    f"[REG-{i}] Source: {r['metadata'].get('filename','?')} "
                    f"(chunk {r['metadata'].get('chunk_idx','?')})\n"
                    f"{r['document']}\n"
                    f"   → Rerank score: {r.get('rerank_score', 0):.3f}"
                )

        return "\n\n".join(sections)


# ══════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":

    retriever = ForensicsRetriever()

    TEST_QUERIES = [
        "show me structuring transactions below the reporting threshold",
        "which accounts are involved in high risk corridors to UAE",
        "what does FATF say about smurfing and aggregation",
        "explain the SAR filing timeline for continuing suspicious activity",
        "find dormant accounts that were suddenly reactivated with large amounts",
    ]

    for query in TEST_QUERIES:
        print(f"\n{'═'*65}")
        print(f"  QUERY: {query}")
        print(f"{'═'*65}")

        results = retriever.retrieve(query, top_k=5, verbose=True)

        print(f"\n📦 FINAL TOP {len(results['all_results'])} RESULTS:")
        for i, r in enumerate(results["all_results"], 1):
            print(
                f"  {i}. [{r['collection']:15s}] "
                f"rerank={r.get('rerank_score', 0):.3f} | "
                f"rrf={r.get('rrf_score', 0):.4f} | "
                f"{r['document'][:80]}..."
            )

        print(f"\n  Breakdown → "
              f"TXN: {len(results['transactions'])} | "
              f"GRAPHS: {len(results['graph_captions'])} | "
              f"REGS: {len(results['regulations'])}")

        print("\n📄 FORMATTED CONTEXT FOR LLM:")
        print(retriever.format_context(results))
        print()
