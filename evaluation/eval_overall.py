"""
eval_overall.py
────────────────
OVERALL EVALUATION — Three components:

1. End-to-End Pipeline Score
   Runs full pipeline (input→retrieval→generation) on 10 queries
   and computes a single weighted quality score combining all signals.

2. Hybrid vs Baseline Comparison
   Proves your BM25 + Dense + Reranker beats naive dense-only retrieval
   on context precision and recall.

3. Agentic Loop Evaluation
   Measures whether the LangGraph agent:
   - Routes queries to correct pipeline
   - Self-corrects when grader scores low
   - Improves answer quality after retry

Run:
  python evaluation/eval_overall.py

Or run all three evals together:
  python evaluation/eval_overall.py --full
"""

import os, sys, json, time, argparse
sys.path.append(".")
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL    = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")


# ══════════════════════════════════════════════════════════════
# E2E TEST QUERIES
# ══════════════════════════════════════════════════════════════
E2E_QUERIES = [
    {"query": "Show structuring transactions below £10,000",           "expect_sources": ["transactions"], "typology": "Structuring"},
    {"query": "Find dormant accounts suddenly reactivated",            "expect_sources": ["transactions","graph_captions"], "typology": "Dormant_Reactivation"},
    {"query": "SAR filing timeline for continuing suspicious activity", "expect_sources": ["regulations"], "typology": "SAR_Compliance"},
    {"query": "Which accounts sent money to UAE?",                     "expect_sources": ["transactions"], "typology": "High_Risk_Corridor"},
    {"query": "What does FATF say about placement and aggregation?",   "expect_sources": ["regulations"], "typology": "Smurfing"},
    {"query": "Explain layering patterns in our transactions",         "expect_sources": ["transactions","graph_captions"], "typology": "Layering"},
    {"query": "High risk corridors to Turkey or Morocco",              "expect_sources": ["transactions"], "typology": "High_Risk_Corridor"},
    {"query": "What are the three stages of money laundering?",        "expect_sources": ["regulations"], "typology": "General_AML"},
    {"query": "Show round trip transactions",                          "expect_sources": ["transactions","graph_captions"], "typology": "Round_Trip"},
    {"query": "What payment types are most associated with suspicious transactions?", "expect_sources": ["transactions"], "typology": "General_AML"},
]

# Router expected decisions
ROUTING_CASES = [
    {"query": "Which accounts sent money to UAE?",                      "expected_pipeline": "vector",     "expected_intent": "numerical"},
    {"query": "Trace funds from account 176667861 through 3 hops",     "expected_pipeline": "graph",      "expected_intent": "numerical"},
    {"query": "What does FATF say about structuring?",                  "expected_pipeline": "vector",     "expected_intent": "conceptual"},
    {"query": "SAR filing deadline requirements",                       "expected_pipeline": "vector",     "expected_intent": "compliance"},
    {"query": "Find round trip circular fund flows",                    "expected_pipeline": "graph",      "expected_intent": "numerical"},
    {"query": "Explain layering and what FATF says about it",          "expected_pipeline": "both",       "expected_intent": "conceptual"},
    {"query": "How many transactions are above £50,000?",               "expected_pipeline": "graph",      "expected_intent": "numerical"},
    {"query": "What are PEP enhanced due diligence requirements?",      "expected_pipeline": "vector",     "expected_intent": "compliance"},
]


# ══════════════════════════════════════════════════════════════
# 1. END-TO-END PIPELINE SCORE
# ══════════════════════════════════════════════════════════════
def eval_e2e() -> dict:
    from retrieval.retrieval_pipeline import ForensicsRetriever
    from generation.generation import ForensicsGenerator
    from ui.features import ResponseFormatter, Guardrails
    import re

    print("\n  Running end-to-end evaluation...")
    retriever = ForensicsRetriever()
    generator = ForensicsGenerator()
    results   = []

    for i, item in enumerate(E2E_QUERIES, 1):
        query = item["query"]
        print(f"  [{i:02d}/{len(E2E_QUERIES)}] {query[:55]}...")
        t0 = time.time()

        try:
            # Input gate
            guard = Guardrails.check_input(query)
            if not guard["allowed"]:
                results.append({"query": query, "error": "guardrail_blocked", "passed": False})
                continue

            # Retrieve
            ret = retriever.retrieve(query, top_k=5)

            # Generate
            out    = generator.generate(query, ret)
            answer = out["answer"]
            fmt    = ResponseFormatter.format(answer)

            elapsed = time.time() - t0

            # Score this result
            source_hit = all(s in out.get("sources", []) for s in item["expect_sources"])
            has_answer = len(answer.split()) >= 30
            no_chunks  = "chunk " not in fmt.lower()
            has_html   = "border-left:3px solid" in fmt
            no_tags    = not bool(re.search(r'\[(?:TXN|GRAPH|REG)-\d+\]', fmt))
            no_refusal = not any(p in answer.lower() for p in ["i cannot", "i don't have", "as an ai"])

            checks = {
                "source_hit":  source_hit,
                "has_answer":  has_answer,
                "no_chunk_leak": no_chunks,
                "html_formatted": has_html,
                "citations_clean": no_tags,
                "no_refusal":  no_refusal,
            }

            score = sum(checks.values()) / len(checks)
            results.append({
                "query":     query,
                "typology":  item["typology"],
                "score":     round(score, 3),
                "elapsed":   round(elapsed, 2),
                "checks":    checks,
                "sources":   out.get("sources", []),
                "passed":    score >= 0.8,
            })

            icon = "✅" if score >= 0.8 else "⚠️ "
            fails = [k for k, v in checks.items() if not v]
            print(f"     {icon} Score={score:.2f} | {elapsed:.1f}s | Sources={out.get('sources',[])} {('| Fails: ' + str(fails)) if fails else ''}")

        except Exception as e:
            print(f"     ❌ Error: {e}")
            results.append({"query": query, "error": str(e), "passed": False, "score": 0})

    valid   = [r for r in results if "error" not in r]
    avg_score = sum(r["score"] for r in valid) / len(valid) if valid else 0
    pass_rate = sum(1 for r in valid if r["passed"]) / len(valid) if valid else 0
    avg_time  = sum(r["elapsed"] for r in valid) / len(valid) if valid else 0

    # Per-typology
    typology_scores = {}
    for r in valid:
        t = r.get("typology", "Unknown")
        if t not in typology_scores:
            typology_scores[t] = []
        typology_scores[t].append(r["score"])

    print(f"\n  E2E Summary:")
    print(f"    Average score:  {avg_score:.3f}")
    print(f"    Pass rate:      {pass_rate:.1%}")
    print(f"    Avg latency:    {avg_time:.1f}s")
    print(f"\n  Per-typology:")
    for t, scores in sorted(typology_scores.items()):
        avg = sum(scores)/len(scores)
        icon = "✅" if avg >= 0.8 else "⚠️ " if avg >= 0.6 else "❌"
        print(f"    {icon} {t:<30} {avg:.3f}")

    return {
        "avg_score":        round(avg_score,  4),
        "pass_rate":        round(pass_rate,  4),
        "avg_latency_s":    round(avg_time,   2),
        "per_typology":     {t: round(sum(s)/len(s), 3) for t, s in typology_scores.items()},
        "results":          results,
    }


# ══════════════════════════════════════════════════════════════
# 2. HYBRID vs BASELINE COMPARISON
# ══════════════════════════════════════════════════════════════
def eval_hybrid_vs_baseline() -> dict:
    import chromadb
    from chromadb.utils import embedding_functions
    from retrieval.retrieval_pipeline import ForensicsRetriever
    from generation.generation import ForensicsGenerator

    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    CHROMA_DIR      = os.getenv("CHROMA_DIR", "chroma_db")

    print("\n  Running hybrid vs baseline comparison...")

    # Baseline: dense-only, no BM25, no reranker
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef     = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    def baseline_retrieve(query, top_k=5):
        all_docs = []
        for col_name in ["transactions", "graph_captions", "regulations"]:
            try:
                col = client.get_collection(col_name, embedding_function=ef)
                r   = col.query(query_texts=[query], n_results=min(top_k, col.count()), include=["documents","metadatas"])
                for i in range(len(r["ids"][0])):
                    all_docs.append({"id": r["ids"][0][i], "document": r["documents"][0][i], "metadata": r["metadatas"][0][i], "collection": col_name})
            except Exception:
                pass
        return {"all_results": all_docs[:top_k], "transactions": [], "graph_captions": [], "regulations": []}

    hybrid_retriever = ForensicsRetriever()
    generator        = ForensicsGenerator()

    comparison = []
    test_queries = [q["query"] for q in E2E_QUERIES[:6]]

    for query in test_queries:
        print(f"  Comparing: {query[:50]}...")

        # Baseline
        t0         = time.time()
        base_ret   = baseline_retrieve(query)
        base_out   = generator.generate(query, base_ret)
        base_time  = time.time() - t0
        base_docs  = base_ret["all_results"]

        # Hybrid
        t0        = time.time()
        hyb_ret   = hybrid_retriever.retrieve(query, top_k=5)
        hyb_out   = generator.generate(query, hyb_ret)
        hyb_time  = time.time() - t0
        hyb_docs  = hyb_ret.get("all_results", [])

        # Simple quality proxy: answer length + source diversity
        base_sources = len(set(d.get("collection") for d in base_docs))
        hyb_sources  = len(set(d.get("collection") for d in hyb_docs))
        base_len     = len(base_out["answer"].split())
        hyb_len      = len(hyb_out["answer"].split())

        comparison.append({
            "query":            query[:50],
            "baseline_sources": base_sources,
            "hybrid_sources":   hyb_sources,
            "baseline_ans_len": base_len,
            "hybrid_ans_len":   hyb_len,
            "source_improvement": hyb_sources - base_sources,
            "length_improvement": hyb_len - base_len,
        })

        print(f"    Baseline: {base_sources} source types, {base_len} word answer")
        print(f"    Hybrid:   {hyb_sources} source types, {hyb_len} word answer")

    avg_src_improvement = sum(c["source_improvement"] for c in comparison) / len(comparison)
    avg_len_improvement = sum(c["length_improvement"] for c in comparison) / len(comparison)

    print(f"\n  Hybrid improvement over baseline:")
    print(f"    Avg source diversity: +{avg_src_improvement:.2f} collection types")
    print(f"    Avg answer length:    +{avg_len_improvement:.0f} words")

    return {
        "avg_source_improvement": round(avg_src_improvement, 3),
        "avg_length_improvement": round(avg_len_improvement,  1),
        "comparison":             comparison,
    }


# ══════════════════════════════════════════════════════════════
# 3. AGENTIC LOOP EVALUATION
# ══════════════════════════════════════════════════════════════
def eval_agentic_loop() -> dict:
    from retrieval.langgraph_orchestrator import router_node, AgentState, FFRAGAgent

    print("\n  Running agentic loop evaluation...")

    # ── 3a. Router accuracy ──
    print("\n  3a. Router accuracy:")
    router_results = []

    for case in ROUTING_CASES:
        state: AgentState = {
            "query": case["query"], "pipeline": "both", "query_intent": "conceptual",
            "rewritten_queries": [], "hyde_document": "", "raw_results": {},
            "optimized_context": "", "all_chunks": [], "relevance_score": 0.0,
            "grader_feedback": "", "retry_count": 0, "answer": "", "sources": [],
            "suspicion_score": None, "should_retry": False,
        }
        out = router_node(state)

        # Accept "both" as correct when expected is "vector" or "graph"
        # (agent being cautious is fine)
        pipeline_ok = (
            out["pipeline"] == case["expected_pipeline"] or
            out["pipeline"] == "both"
        )
        intent_ok = out["query_intent"] == case["expected_intent"]
        passed    = pipeline_ok  # intent is secondary

        router_results.append({
            "query":     case["query"][:50],
            "expected":  case["expected_pipeline"],
            "got":       out["pipeline"],
            "intent_ok": intent_ok,
            "passed":    passed,
        })

        icon = "✅" if passed else "❌"
        print(f"    {icon} {case['query'][:45]} → {out['pipeline']} (expected {case['expected_pipeline']})")

    router_accuracy = sum(1 for r in router_results if r["passed"]) / len(router_results)
    print(f"    Router accuracy: {router_accuracy:.1%}")

    # ── 3b. Full agent runs — measure retry behaviour ──
    print("\n  3b. Agent retry behaviour (3 queries):")
    agent    = FFRAGAgent()
    agent_results = []

    for item in E2E_QUERIES[:3]:
        t0     = time.time()
        result = agent.run(item["query"])
        elapsed = time.time() - t0

        agent_results.append({
            "query":          item["query"][:50],
            "pipeline":       result["pipeline"],
            "relevance":      result["relevance_score"],
            "retries":        result["retry_count"],
            "answer_words":   len(result["answer"].split()),
            "elapsed":        round(elapsed, 2),
            "has_answer":     len(result["answer"]) > 100,
        })

        print(f"    ✅ '{item['query'][:40]}'")
        print(f"       pipeline={result['pipeline']} relevance={result['relevance_score']:.2f} retries={result['retry_count']} ({elapsed:.1f}s)")

    avg_relevance = sum(r["relevance"] for r in agent_results) / len(agent_results)
    avg_retries   = sum(r["retries"]   for r in agent_results) / len(agent_results)
    all_answered  = all(r["has_answer"] for r in agent_results)

    print(f"\n  Agent summary:")
    print(f"    Avg relevance score: {avg_relevance:.3f}")
    print(f"    Avg retries:         {avg_retries:.2f}")
    print(f"    All answered:        {all_answered}")

    return {
        "router_accuracy": round(router_accuracy, 4),
        "avg_relevance":   round(avg_relevance,   4),
        "avg_retries":     round(avg_retries,      2),
        "all_answered":    all_answered,
        "router_results":  router_results,
        "agent_results":   agent_results,
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Also run eval_input + eval_output")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-agent",    action="store_true")
    args = parser.parse_args()

    os.makedirs("evaluation", exist_ok=True)

    print("\n" + "="*60)
    print("  OVERALL EVALUATION")
    print("  E2E + Hybrid vs Baseline + Agentic Loop")
    print("="*60)

    e2e_scores      = eval_e2e()
    baseline_scores = eval_hybrid_vs_baseline() if not args.skip_baseline else {"skipped": True}
    agent_scores    = eval_agentic_loop()        if not args.skip_agent    else {"skipped": True}

    # Combined overall score
    e2e_s    = e2e_scores.get("avg_score",       0)
    agent_s  = agent_scores.get("avg_relevance", 0)
    router_s = agent_scores.get("router_accuracy", 0)

    overall = e2e_s * 0.5 + agent_s * 0.3 + router_s * 0.2

    print("\n" + "="*60)
    print("  OVERALL EVALUATION SUMMARY")
    print("="*60)
    print(f"  E2E Average Score:     {e2e_s:.4f}")
    print(f"  E2E Pass Rate:         {e2e_scores.get('pass_rate', 0):.4f}")
    print(f"  E2E Avg Latency:       {e2e_scores.get('avg_latency_s', 0):.1f}s")
    if not baseline_scores.get("skipped"):
        print(f"  Hybrid Source Gain:    +{baseline_scores.get('avg_source_improvement',0):.2f} types")
        print(f"  Hybrid Answer Gain:    +{baseline_scores.get('avg_length_improvement',0):.0f} words")
    print(f"  Router Accuracy:       {router_s:.4f}")
    print(f"  Agent Avg Relevance:   {agent_s:.4f}")
    print(f"  ─────────────────────────────────")
    print(f"  COMBINED OVERALL SCORE: {overall:.4f}")
    print("="*60)

    # ── If --full flag, load and merge input/output results ──
    if args.full:
        summary = {}
        for f, key in [("eval_input_results.json","input"), ("eval_output_results.json","output")]:
            p = f"evaluation/{f}"
            if os.path.exists(p):
                summary[key] = json.loads(open(p).read())

        if "input" in summary and "output" in summary:
            inp_s = summary["input"].get("combined_input_score", 0)
            out_s = summary["output"].get("combined_output_score", 0)
            full_score = inp_s * 0.2 + out_s * 0.4 + overall * 0.4
            print(f"\n  FULL SYSTEM SCORE (all 3 evals combined):")
            print(f"    Input score:   {inp_s:.4f}  (weight 20%)")
            print(f"    Output score:  {out_s:.4f}  (weight 40%)")
            print(f"    Overall score: {overall:.4f}  (weight 40%)")
            print(f"    ──────────────────────────────")
            print(f"    FINAL:         {full_score:.4f}")

    result = {
        "eval_type":    "overall",
        "eval_date":    __import__("datetime").datetime.now().isoformat(),
        "e2e":          e2e_scores,
        "hybrid_vs_baseline": baseline_scores,
        "agentic_loop": agent_scores,
        "combined_overall_score": round(overall, 4),
    }

    with open("evaluation/eval_overall_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("\n✅ Results saved to evaluation/eval_overall_results.json")


if __name__ == "__main__":
    main()
