"""
eval_retrieval_metrics.py
──────────────────────────
Computes Precision, Recall, and MRR at the retrieval level.

  Precision@K  = relevant retrieved / K
  Recall@K     = relevant retrieved / total relevant
  MRR          = mean(1 / rank_of_first_relevant)

Ground truth is defined per query as:
  - expected_collection : which ChromaDB collection should be hit
  - expected_keywords   : words that must appear in a relevant chunk
  - expected_typology   : typology tag that marks a chunk as relevant

Also compares Hybrid (BM25 + Dense + Reranker) vs Baseline (Dense only)
so you have a before/after MRR to show evaluators.

Run:
  python evaluation/eval_retrieval_metrics.py

Results saved to:
  evaluation/retrieval_metrics.json
"""


import os, sys, json, time
sys.path.append(".")

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


# ══════════════════════════════════════════════════════════════
# GROUND TRUTH DATASET
# Each query has a definition of what a "relevant" chunk looks like
# ══════════════════════════════════════════════════════════════
RETRIEVAL_GROUND_TRUTH = [
    {
        "query": "Show structuring transactions below £10,000",
        "relevant_collection": "transactions",
        "relevant_keywords":   ["structuring", "suspicious"],
        "relevant_typology":   "Structuring",
        "k": 5,
    },
    {
        "query": "SAR filing timeline continuing suspicious activity",
        "relevant_collection": "regulations",
        "relevant_keywords":   ["SAR", "filing", "30", "90", "120"],
        "relevant_keywords_any": True,  # any keyword match counts
        "k": 5,
    },
    {
        "query": "Find dormant accounts suddenly reactivated",
        "relevant_collection": "transactions",
        "relevant_keywords":   ["dormant", "reactivat"],
        "relevant_typology":   "Dormant_Reactivation",
        "k": 5,
    },
    {
        "query": "FATF placement aggregation smurfing",
        "relevant_collection": "regulations",
        "relevant_keywords":   ["placement", "aggregat", "layering", "structur"],
        "relevant_keywords_any": True,
        "k": 5,
    },
    {
        "query": "Which accounts sent money to UAE high risk corridor",
        "relevant_collection": "transactions",
        "relevant_keywords":   ["UAE", "high_risk", "suspicious"],
        "relevant_typology":   "High_Risk_Corridor",
        "k": 5,
    },
    {
        "query": "Layering transactions network graph analysis",
        "relevant_collection": "graph_captions",
        "relevant_keywords":   ["layering", "chain", "sequential"],
        "k": 5,
    },
    {
        "query": "Round trip transactions circular fund flow",
        "relevant_collection": "transactions",
        "relevant_keywords":   ["round_trip", "Round_Trip", "suspicious"],
        "relevant_typology":   "Round_Trip",
        "k": 5,
    },
    {
        "query": "Currency mismatch payment received currency",
        "relevant_collection": "transactions",
        "relevant_keywords":   ["currency", "mismatch", "suspicious"],
        "relevant_typology":   "Currency_Mismatch",
        "k": 5,
    },
    {
        "query": "FATF recommendations enhanced due diligence PEP",
        "relevant_collection": "regulations",
        "relevant_keywords":   ["due diligence", "PEP", "politically exposed", "enhanced"],
        "relevant_keywords_any": True,
        "k": 5,
    },
    {
        "query": "Rapid succession high frequency transactions",
        "relevant_collection": "transactions",
        "relevant_keywords":   ["rapid", "succession", "suspicious"],
        "relevant_typology":   "Rapid_Succession",
        "k": 5,
    },
]


# ══════════════════════════════════════════════════════════════
# RELEVANCE JUDGE
# ══════════════════════════════════════════════════════════════
def is_relevant(chunk: dict, ground_truth: dict) -> bool:
    """
    Decide if a retrieved chunk is relevant for a query.
    A chunk is relevant if it:
      1. Comes from the expected collection, AND
      2. Contains at least one expected keyword (case-insensitive)
         OR matches the expected typology in metadata
    """
    doc      = (chunk.get("document") or chunk.get("text") or "").lower()
    meta     = chunk.get("metadata", {})
    col      = chunk.get("collection", "")

    # Collection check
    if col != ground_truth["relevant_collection"]:
        return False

    # Typology check (fast path for transactions)
    expected_typology = ground_truth.get("relevant_typology", "")
    if expected_typology:
        meta_typology = str(meta.get("typology", "")).lower()
        if expected_typology.lower() in meta_typology:
            return True

    # Keyword check
    keywords = ground_truth.get("relevant_keywords", [])
    any_match = ground_truth.get("relevant_keywords_any", False)

    if any_match:
        return any(kw.lower() in doc for kw in keywords)
    else:
        return all(kw.lower() in doc for kw in keywords)


# ══════════════════════════════════════════════════════════════
# METRICS COMPUTATION
# ══════════════════════════════════════════════════════════════
def precision_at_k(retrieved: list[dict], ground_truth: dict) -> float:
    """Precision@K = relevant retrieved / K"""
    k         = ground_truth["k"]
    top_k     = retrieved[:k]
    n_relevant = sum(1 for c in top_k if is_relevant(c, ground_truth))
    return n_relevant / k if k > 0 else 0.0


def recall_at_k(retrieved: list[dict], ground_truth: dict, total_relevant: int) -> float:
    """Recall@K = relevant retrieved / total relevant in corpus"""
    if total_relevant == 0:
        return 0.0
    k          = ground_truth["k"]
    top_k      = retrieved[:k]
    n_retrieved = sum(1 for c in top_k if is_relevant(c, ground_truth))
    return n_retrieved / total_relevant


def reciprocal_rank(retrieved: list[dict], ground_truth: dict) -> float:
    """
    Reciprocal Rank = 1 / rank_of_first_relevant_chunk
    If no relevant chunk found, RR = 0
    """
    for rank, chunk in enumerate(retrieved, start=1):
        if is_relevant(chunk, ground_truth):
            return 1.0 / rank
    return 0.0


def count_relevant_in_corpus(collection_name: str, ground_truth: dict) -> int:
    """Count how many relevant docs exist in the full ChromaDB collection."""
    try:
        import chromadb
        from chromadb.utils import embedding_functions

        CHROMA_DIR      = os.getenv("CHROMA_DIR", "chroma_db")
        EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

        client = chromadb.PersistentClient(path=CHROMA_DIR)
        ef     = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        col    = client.get_collection(collection_name, embedding_function=ef)

        # Fetch all docs (capped at 500 for speed)
        result = col.get(limit=500, include=["documents","metadatas"])
        count  = 0

        for i in range(len(result["ids"])):
            chunk = {
                "document":   result["documents"][i],
                "metadata":   result["metadatas"][i],
                "collection": collection_name,
            }
            if is_relevant(chunk, ground_truth):
                count += 1

        return max(count, 1)  # avoid div/0

    except Exception:
        return 10  # safe fallback


# ══════════════════════════════════════════════════════════════
# BASELINE RETRIEVER
# ══════════════════════════════════════════════════════════════
def baseline_retrieve(query: str, k: int = 5) -> list[dict]:
    """Dense-only retrieval — no BM25, no reranker."""
    import chromadb
    from chromadb.utils import embedding_functions

    CHROMA_DIR      = os.getenv("CHROMA_DIR", "chroma_db")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef     = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    all_docs = []
    for col_name in ["transactions", "graph_captions", "regulations"]:
        try:
            col    = client.get_collection(col_name, embedding_function=ef)
            result = col.query(
                query_texts=[query],
                n_results=min(k, col.count()),
                include=["documents", "metadatas"],
            )
            for i in range(len(result["ids"][0])):
                all_docs.append({
                    "id":         result["ids"][0][i],
                    "document":   result["documents"][0][i],
                    "metadata":   result["metadatas"][0][i],
                    "collection": col_name,
                })
        except Exception:
            pass

    return all_docs[:k]


# ══════════════════════════════════════════════════════════════
# MAIN EVALUATOR
# ══════════════════════════════════════════════════════════════
def evaluate_retrieval(retriever_fn, label: str) -> dict:
    """
    Run precision, recall, MRR evaluation for a given retriever function.

    Args:
        retriever_fn : callable(query, k) → list[dict]
        label        : "Hybrid" or "Baseline"
    """
    print(f"\n  Evaluating {label} retriever ({len(RETRIEVAL_GROUND_TRUTH)} queries)...")

    precisions = []
    recalls    = []
    rrs        = []
    per_query  = []

    print(f"\n  {'Query':<45} {'P@K':>6} {'R@K':>6} {'RR':>6}")
    print(f"  {'─'*65}")

    for gt in RETRIEVAL_GROUND_TRUTH:
        query = gt["query"]
        k     = gt["k"]

        # Retrieve
        retrieved = retriever_fn(query, k)

        # Count relevant in corpus for recall denominator
        total_relevant = count_relevant_in_corpus(gt["relevant_collection"], gt)

        # Compute metrics
        p  = precision_at_k(retrieved, gt)
        r  = recall_at_k(retrieved, gt, total_relevant)
        rr = reciprocal_rank(retrieved, gt)

        precisions.append(p)
        recalls.append(r)
        rrs.append(rr)

        per_query.append({
            "query":           query,
            "collection":      gt["relevant_collection"],
            "precision_at_k":  round(p,  4),
            "recall_at_k":     round(r,  4),
            "reciprocal_rank": round(rr, 4),
            "total_relevant":  total_relevant,
            "k":               k,
        })

        print(f"  {query[:43]:<45} {p:>6.3f} {r:>6.3f} {rr:>6.3f}")

    # Aggregate
    mean_precision = sum(precisions) / len(precisions)
    mean_recall    = sum(recalls)    / len(recalls)
    mrr            = sum(rrs)        / len(rrs)

    print(f"  {'─'*65}")
    print(f"  {'MEAN':<45} {mean_precision:>6.3f} {mean_recall:>6.3f} {mrr:>6.3f}")

    return {
        "label":          label,
        "mean_precision": round(mean_precision, 4),
        "mean_recall":    round(mean_recall,    4),
        "mrr":            round(mrr,            4),
        "per_query":      per_query,
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    from retrieval.retrieval_pipeline import ForensicsRetriever

    os.makedirs("evaluation", exist_ok=True)

    print("\n" + "="*65)
    print("  RETRIEVAL METRICS EVALUATION")
    print("  Precision@K · Recall@K · MRR")
    print("="*65)

    # ── Hybrid pipeline ──
    hybrid_retriever = ForensicsRetriever()

    def hybrid_fn(query, k):
        results = hybrid_retriever.retrieve(query, top_k=k)
        chunks  = results.get("all_results", [])
        # Ensure collection tag is set
        for col in ["transactions", "graph_captions", "regulations"]:
            for c in results.get(col, []):
                c["collection"] = col
        return chunks[:k]

    hybrid_scores = evaluate_retrieval(hybrid_fn, "Hybrid")

    # ── Baseline pipeline ──
    print()
    baseline_scores = evaluate_retrieval(baseline_retrieve, "Baseline")

    # ── Delta ──
    p_delta  = hybrid_scores["mean_precision"] - baseline_scores["mean_precision"]
    r_delta  = hybrid_scores["mean_recall"]    - baseline_scores["mean_recall"]
    mrr_delta= hybrid_scores["mrr"]            - baseline_scores["mrr"]

    print("\n" + "="*65)
    print("  FINAL RESULTS")
    print("="*65)
    print(f"\n  {'Metric':<20} {'Baseline':>10} {'Hybrid':>10} {'Delta':>10}")
    print(f"  {'─'*52}")
    print(f"  {'Precision@K':<20} {baseline_scores['mean_precision']:>10.4f} {hybrid_scores['mean_precision']:>10.4f} {'↑' if p_delta>0 else '↓'}{abs(p_delta):>8.4f}")
    print(f"  {'Recall@K':<20} {baseline_scores['mean_recall']:>10.4f} {hybrid_scores['mean_recall']:>10.4f} {'↑' if r_delta>0 else '↓'}{abs(r_delta):>8.4f}")
    print(f"  {'MRR':<20} {baseline_scores['mrr']:>10.4f} {hybrid_scores['mrr']:>10.4f} {'↑' if mrr_delta>0 else '↓'}{abs(mrr_delta):>8.4f}")
    print(f"  {'─'*52}")

    # Highlight MRR interpretation
    if hybrid_scores["mrr"] > 0:
        avg_rank = 1.0 / hybrid_scores["mrr"]
        print(f"\n  MRR={hybrid_scores['mrr']:.4f} → first relevant chunk appears at avg rank {avg_rank:.1f}")
    if hybrid_scores["mrr"] >= 0.9:
        print("  ✅ Excellent — first relevant result is almost always rank 1")
    elif hybrid_scores["mrr"] >= 0.7:
        print("  ✅ Good — first relevant result typically within top 2")
    elif hybrid_scores["mrr"] >= 0.5:
        print("  ⚠️  Fair — first relevant result within top 3 on average")
    else:
        print("  ❌ Poor — relevant results buried in rankings")

    print("="*65)

    # Save
    output = {
        "eval_type":  "retrieval_metrics",
        "eval_date":  __import__("datetime").datetime.now().isoformat(),
        "n_queries":  len(RETRIEVAL_GROUND_TRUTH),
        "hybrid":     hybrid_scores,
        "baseline":   baseline_scores,
        "delta": {
            "precision": round(p_delta,   4),
            "recall":    round(r_delta,   4),
            "mrr":       round(mrr_delta, 4),
        },
    }

    with open("evaluation/retrieval_metrics.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print("\n✅ Saved to evaluation/retrieval_metrics.json\n")


if __name__ == "__main__":
    main()
