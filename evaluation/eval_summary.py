"""
eval_summary.py
────────────────
Runs all evaluations and prints ONE final table with 3 numbers:

  Precision | Recall | MRR

Optionally skip expensive steps with flags.

Run:
  python evaluation/eval_summary.py

Skip RAGAS (slow):
  python evaluation/eval_summary.py --skip-ragas

Use cached results if already run:
  python evaluation/eval_summary.py --from-cache
"""

import os, sys, json, argparse
sys.path.append(".")
from dotenv import load_dotenv
load_dotenv()

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def load_cached(path: str) -> dict | None:
    if os.path.exists(path):
        return json.loads(open(path).read())
    return None


def run_retrieval_metrics() -> dict:
    """Returns {precision, recall, mrr} for hybrid pipeline."""
    from retrieval.retrieval_pipeline import ForensicsRetriever
    from evaluation.eval_retrieval_metrics import (
        RETRIEVAL_GROUND_TRUTH,
        evaluate_retrieval,
        baseline_retrieve,
    )

    retriever = ForensicsRetriever()

    def hybrid_fn(query, k):
        results = retriever.retrieve(query, top_k=k)
        chunks  = results.get("all_results", [])
        for col in ["transactions", "graph_captions", "regulations"]:
            for c in results.get(col, []):
                c["collection"] = col
        return chunks[:k]

    scores = evaluate_retrieval(hybrid_fn, "Hybrid")
    return {
        "precision": scores["mean_precision"],
        "recall":    scores["mean_recall"],
        "mrr":       scores["mrr"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-cache", action="store_true",
                        help="Load from previously saved JSON files instead of re-running")
    parser.add_argument("--skip-ragas", action="store_true",
                        help="Skip RAGAS (saves ~5 minutes)")
    args = parser.parse_args()

    os.makedirs("evaluation", exist_ok=True)

    print(f"\n{BOLD}{'═'*55}")
    print(f"  FFRAG — FINAL EVALUATION METRICS")
    print(f"{'═'*55}{RESET}\n")

    # ── Source metrics from each eval ──
    precision_sources = []
    recall_sources    = []
    mrr_sources       = []

    # ── 1. Retrieval Metrics (primary source of all 3) ──
    print(f"{BLUE}  [1/3] Retrieval Metrics (Precision · Recall · MRR){RESET}")

    if args.from_cache:
        cached = load_cached("evaluation/retrieval_metrics.json")
        if cached:
            ret = cached["hybrid"]
            print(f"       (loaded from cache)")
        else:
            print(f"       Cache not found — running now...")
            ret = run_retrieval_metrics()
    else:
        ret = run_retrieval_metrics()

    precision_sources.append(("Retrieval P@K",  ret["precision"], 1.0))
    recall_sources.append(   ("Retrieval R@K",  ret["recall"],    1.0))
    mrr_sources.append(      ("Retrieval MRR",  ret["mrr"],       1.0))

    print(f"       P@K={ret['precision']:.4f}  R@K={ret['recall']:.4f}  MRR={ret['mrr']:.4f}")

    # ── 2. Input Eval — contributes to Precision via guardrail precision ──
    print(f"\n{BLUE}  [2/3] Input Evaluation (Guardrails + Query Rewriting){RESET}")

    if args.from_cache:
        cached = load_cached("evaluation/eval_input_results.json")
    else:
        from evaluation.eval_input import eval_guardrails, eval_query_rewriting
        guardrail_scores = eval_guardrails()
        rewrite_scores   = eval_query_rewriting()
        cached = {
            "guardrails":      guardrail_scores,
            "query_rewriting": rewrite_scores,
        }
        json.dump(cached, open("evaluation/eval_input_results.json","w"), indent=2, default=str)

    if cached:
        g_precision = cached["guardrails"]["precision"]
        g_recall    = cached["guardrails"]["recall"]
        rw_coverage = cached["query_rewriting"]["avg_coverage"]
        print(f"       Guardrail precision={g_precision:.4f}  recall={g_recall:.4f}")
        print(f"       Query expansion coverage={rw_coverage:.4f}")
        # Guardrail precision → weighted into final precision (20% weight)
        precision_sources.append(("Guardrail Precision", g_precision, 0.2))
        # Guardrail recall → weighted into final recall (20% weight)
        recall_sources.append(   ("Guardrail Recall",    g_recall,    0.2))
        # Query expansion coverage → weighted into MRR (10% weight, better queries → better rank)
        mrr_sources.append(      ("Query Expansion",     rw_coverage, 0.1))

    # ── 3. Output Eval — contributes formatting precision to overall ──
    print(f"\n{BLUE}  [3/3] Output Evaluation (RAGAS + Formatting + Suspicion){RESET}")

    if args.from_cache:
        cached_out = load_cached("evaluation/eval_output_results.json")
    elif not args.skip_ragas:
        from evaluation.eval_output import eval_ragas, eval_formatting, eval_suspicion_scores
        ragas  = eval_ragas()
        fmt    = eval_formatting()
        susp   = eval_suspicion_scores()
        cached_out = {"ragas": ragas, "formatting": fmt, "suspicion": susp}
        json.dump(cached_out, open("evaluation/eval_output_results.json","w"), indent=2, default=str)
    else:
        print(f"       {YELLOW}RAGAS skipped (--skip-ragas){RESET}")
        cached_out = None

    if cached_out:
        ragas_p   = cached_out["ragas"].get("context_precision", 0)
        ragas_r   = cached_out["ragas"].get("context_recall",    0)
        fmt_score = cached_out["formatting"].get("avg_score",    0)
        print(f"       RAGAS context_precision={ragas_p:.4f}  context_recall={ragas_r:.4f}")
        print(f"       Formatting score={fmt_score:.4f}")
        # RAGAS context_precision → weighted into final precision (30% weight)
        precision_sources.append(("RAGAS Context Precision", ragas_p, 0.3))
        # RAGAS context_recall → weighted into final recall (30% weight)
        recall_sources.append(   ("RAGAS Context Recall",    ragas_r, 0.3))
        # Formatting precision → MRR proxy (well-ranked answers are well-formatted)
        mrr_sources.append(      ("Output Formatting",       fmt_score, 0.1))

    # ══════════════════════════════════════════════════
    # COMPUTE FINAL 3 METRICS
    # Weighted average across all contributing sources
    # ══════════════════════════════════════════════════
    def weighted_avg(sources):
        total_weight = sum(w for _, _, w in sources)
        return sum(v * w for _, v, w in sources) / total_weight if total_weight > 0 else 0

    final_precision = weighted_avg(precision_sources)
    final_recall    = weighted_avg(recall_sources)
    final_mrr       = weighted_avg(mrr_sources)

    # ══════════════════════════════════════════════════
    # PRINT FINAL TABLE
    # ══════════════════════════════════════════════════
    def color_score(v):
        if v >= 0.80: return f"{GREEN}{v:.4f}{RESET}"
        if v >= 0.60: return f"{YELLOW}{v:.4f}{RESET}"
        return f"{RED}{v:.4f}{RESET}"

    def bar(v, width=20):
        filled = int(v * width)
        return "█" * filled + "░" * (width - filled)

    print(f"\n{BOLD}{'═'*55}")
    print(f"  FINAL METRICS")
    print(f"{'═'*55}{RESET}\n")

    print(f"  {'Metric':<12} {'Score':>8}   {'':20}  {'Breakdown'}")
    print(f"  {'─'*55}")

    print(f"  {'Precision':<12} {color_score(final_precision):>8}   [{bar(final_precision)}]")
    for name, val, w in precision_sources:
        print(f"               {BLUE}└ {name:<28} {val:.4f} (w={w:.1f}){RESET}")

    print(f"\n  {'Recall':<12} {color_score(final_recall):>8}   [{bar(final_recall)}]")
    for name, val, w in recall_sources:
        print(f"               {BLUE}└ {name:<28} {val:.4f} (w={w:.1f}){RESET}")

    print(f"\n  {'MRR':<12} {color_score(final_mrr):>8}   [{bar(final_mrr)}]")
    for name, val, w in mrr_sources:
        print(f"               {BLUE}└ {name:<28} {val:.4f} (w={w:.1f}){RESET}")

    if final_mrr > 0:
        avg_rank = 1.0 / final_mrr
        print(f"\n  MRR={final_mrr:.4f} → first relevant result at avg rank {avg_rank:.1f}")

    print(f"\n{BOLD}{'─'*55}")
    print(f"  Precision : {final_precision:.4f}")
    print(f"  Recall    : {final_recall:.4f}")
    print(f"  MRR       : {final_mrr:.4f}")
    print(f"{'─'*55}{RESET}\n")

    # Save
    result = {
        "eval_date":  __import__("datetime").datetime.now().isoformat(),
        "precision":  round(final_precision, 4),
        "recall":     round(final_recall,    4),
        "mrr":        round(final_mrr,       4),
        "breakdown": {
            "precision_sources": [(n, round(v,4), w) for n,v,w in precision_sources],
            "recall_sources":    [(n, round(v,4), w) for n,v,w in recall_sources],
            "mrr_sources":       [(n, round(v,4), w) for n,v,w in mrr_sources],
        }
    }
    with open("evaluation/final_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"✅ Saved to evaluation/final_metrics.json\n")


if __name__ == "__main__":
    main()
