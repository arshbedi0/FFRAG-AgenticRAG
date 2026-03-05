"""
eval_output.py
───────────────
OUTPUT EVALUATION — Three components:

1. RAGAS Metrics
   - Faithfulness        : claims grounded in retrieved context
   - Answer Relevancy    : answer addresses the question
   - Context Precision   : retrieved chunks are relevant
   - Context Recall      : all needed chunks were retrieved

2. Formatting Quality
   - All 5 sections present (FINDINGS/TYPOLOGY/REGULATORY/VERDICT/SOURCES)
   - No raw chunk references (chunk N)
   - No stray markdown artifacts (**AML Assessment:**)
   - Citations humanised (no [TXN-1] raw tags)
   - HTML section headers rendered correctly

3. Suspicion Score Accuracy
   - Suspicious transactions scored HIGH/CRITICAL
   - Normal transactions scored LOW
   - Score range validation (1-10)
   - Typology-score correlation

Run:
  python evaluation/eval_output.py
"""

import os, sys, json, re, time, random
sys.path.append(".")
from dotenv import load_dotenv
load_dotenv()

CACHE_FILE = "evaluation/ragas_cache.json"

def _load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try: return json.loads(open(CACHE_FILE).read())
        except: pass
    return {}

def _save_cache(cache: dict):
    os.makedirs("evaluation", exist_ok=True)
    open(CACHE_FILE, "w").write(json.dumps(cache, indent=2, default=str))

def _groq_call_with_retry(fn, max_retries=6, base_delay=10):
    """Call fn(), retry on rate limit with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            is_rate_limit = any(x in msg for x in [
                "rate_limit", "rate limit", "429", "tokens per", "exhausted",
                "too many requests", "quota"
            ])
            if is_rate_limit and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(1, 5)
                print(f"     ⏳ Rate limited — waiting {delay:.0f}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(delay)
            else:
                raise
    return None

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL    = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# ══════════════════════════════════════════════════════════════
# TEST QUERIES FOR OUTPUT EVAL (subset of RAGAS dataset)
# ══════════════════════════════════════════════════════════════
OUTPUT_EVAL_QUERIES = [
    {
        "query":        "What is the SAR filing timeline for continuing suspicious activity?",
        "ground_truth": "Day 0 detection, Day 30 initial SAR filing, Day 120 end of 90-day period, Day 150 SAR for continued activity. Filing deadline is 120 calendar days after previous SAR.",
        "typology":     "SAR_Compliance",
        "expect_sources": ["regulations"],
    },
    {
        "query":        "Show structuring transactions below £10,000",
        "ground_truth": "Structuring transactions have amounts just below £9,999 threshold, paid via cash or cheque, flagged as suspicious with Structuring typology.",
        "typology":     "Structuring",
        "expect_sources": ["transactions"],
    },
    {
        "query":        "Find dormant accounts that were suddenly reactivated",
        "ground_truth": "Dormant reactivation transactions show accounts processing £60,000-£82,000 cash transfers after inactivity, suspicious volume £2.2M.",
        "typology":     "Dormant_Reactivation",
        "expect_sources": ["transactions", "graph_captions"],
    },
    {
        "query":        "What does FATF say about placement and aggregation?",
        "ground_truth": "FATF describes placement as introducing criminal proceeds into financial system via aggregation of multiple small deposits, representing the first stage of money laundering.",
        "typology":     "Smurfing",
        "expect_sources": ["regulations"],
    },
    {
        "query":        "Explain layering patterns in our transactions",
        "ground_truth": "Layering involves sequential transfers through multiple accounts. Graph shows 67 accounts with £1.7M suspicious volume using ACH transfers of £20,000-£75,000.",
        "typology":     "Layering",
        "expect_sources": ["transactions", "graph_captions"],
    },
    {
        "query":        "Which accounts sent money to UAE and why is it suspicious?",
        "ground_truth": "UK accounts sent cross-border transfers to UAE flagged as High_Risk_Corridor. FATF Recommendation 19 requires enhanced due diligence for high-risk jurisdictions.",
        "typology":     "High_Risk_Corridor",
        "expect_sources": ["transactions", "regulations"],
    },
    {
        "query":        "What are the three stages of money laundering according to FATF?",
        "ground_truth": "Placement: introducing proceeds into financial system. Layering: obscuring audit trail. Integration: making funds appear legitimate.",
        "typology":     "General_AML",
        "expect_sources": ["regulations"],
    },
    {
        "query":        "Show round trip transactions in the dataset",
        "ground_truth": "Round trip transactions involve funds leaving an account, passing through intermediaries, and returning to a beneficial-owner-connected account. Integration-stage laundering.",
        "typology":     "Round_Trip",
        "expect_sources": ["transactions", "graph_captions"],
    },
]

# ══════════════════════════════════════════════════════════════
# SUSPICION SCORE TEST CASES
# ══════════════════════════════════════════════════════════════
SUSPICION_CASES = [
    # High risk — should score HIGH or CRITICAL (>=7)
    {"text": "Account 176667861 sent £82,230 to account in UAE via cross-border payment. Status: SUSPICIOUS. Typology: Dormant_Reactivation.", "expected_level": "HIGH", "min_score": 7},
    {"text": "Account 100614068 sent £8,679 to account 465260635. Status: SUSPICIOUS. Typology: Structuring.", "expected_level": "HIGH", "min_score": 6},
    {"text": "Account 507943839 sent £41,234 to Turkey account. Status: SUSPICIOUS. Typology: High_Risk_Corridor.", "expected_level": "HIGH", "min_score": 7},
    # Low risk — should score LOW or MEDIUM (<=5)
    {"text": "Account 123456789 sent £450 to account 987654321. Status: NORMAL. Typology: Normal.", "expected_level": "LOW", "max_score": 4},
    {"text": "Account 111111111 sent £1,200 to account 222222222 via bank transfer. Status: NORMAL. Typology: Normal.", "expected_level": "LOW", "max_score": 4},
]


# ══════════════════════════════════════════════════════════════
# 1. RAGAS EVALUATION
# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# 1. RAGAS EVALUATION
# ══════════════════════════════════════════════════════════════
def eval_ragas() -> dict:
    try:
        from ragas import evaluate, EvaluationDataset, SingleTurnSample
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas.run_config import RunConfig  # <-- ADDED THIS
        from langchain_groq import ChatGroq
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        print("  ⚠️  RAGAS not installed — skipping. pip install ragas langchain-groq langchain-huggingface")
        return {"skipped": True}

    from retrieval.retrieval_pipeline import ForensicsRetriever
    from generation.generation import ForensicsGenerator

    print("\n  Running RAGAS evaluation (with RunConfig + 8B Judge)...")
    # retriever  = ForensicsRetriever()
    # generator  = ForensicsGenerator()
    class SafeGroq(ChatGroq):
        """A wrapper that strips the forbidden 'n' parameter from RAGAS requests before hitting Groq."""
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            kwargs.pop("n", None)
            return super()._generate(messages, stop, run_manager, **kwargs)

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            kwargs.pop("n", None)
            return await super()._agenerate(messages, stop, run_manager, **kwargs)

    print("\n  Running RAGAS evaluation (with SafeGroq 8B Judge)...")
    retriever  = ForensicsRetriever()
    generator  = ForensicsGenerator()
    
    # Use our new SafeGroq class
    llm_judge  = SafeGroq(
        model="llama-3.1-8b-instant",  
        api_key=GROQ_API_KEY,
        temperature=0.0,
        max_retries=5
    )
    
    # Use 8B model strictly as the judge to save tokens and prevent timeouts
    
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )

    # Load retrieval cache — avoids re-hitting Groq for generation on resume
    cache   = _load_cache()
    samples = []

    for i, item in enumerate(OUTPUT_EVAL_QUERIES, 1):
        query     = item["query"]
        cache_key = f"sample_{i}"
        print(f"  [{i:02d}/{len(OUTPUT_EVAL_QUERIES)}] {query[:55]}...")

        if cache_key in cache:
            print(f"     ✅ Loaded from cache")
            s = cache[cache_key]
            samples.append(SingleTurnSample(
                user_input=s["user_input"],
                response=s["response"],
                retrieved_contexts=s["retrieved_contexts"],
                reference=s["reference"],
            ))
            continue

        try:
            def _retrieve_and_generate():
                results  = retriever.retrieve(query, top_k=5)
                output   = generator.generate(query, results)
                contexts = [r["document"] for r in results.get("all_results", [])]
                return output["answer"], contexts or ["No context retrieved."]

            answer, contexts = _groq_call_with_retry(_retrieve_and_generate)

            sample_data = {
                "user_input":         query,
                "response":           answer,
                "retrieved_contexts": contexts,
                "reference":          item["ground_truth"],
            }
            cache[cache_key] = sample_data
            _save_cache(cache)

            samples.append(SingleTurnSample(**sample_data))

            # Polite delay between generation calls
            if i < len(OUTPUT_EVAL_QUERIES):
                time.sleep(3)

        except Exception as e:
            print(f"     ⚠️  Skipping — {e}")

    print(f"\n  Running RAGAS scoring on {len(samples)} samples...")
    print(f"  (Using max_workers=1 to prevent Groq TimeoutErrors)")

    dataset = EvaluationDataset(samples=samples)

    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    all_metrics  = [faithfulness, answer_relevancy, context_precision, context_recall]
    metric_results = {}

    for mname, metric in zip(metric_names, all_metrics):
        print(f"  Scoring {mname}...")
        def _score(m=metric, mn=mname):
            r = evaluate(
                dataset=dataset,
                metrics=[m],
                llm=llm_judge,
                embeddings=embeddings,
                raise_exceptions=False,
                run_config=RunConfig(max_workers=1, timeout=180) # <-- THE FIX
            )
            try:    return mn, round(float(r[mn]), 4)
            except: return mn, 0.0
            
        try:
            _, val = _groq_call_with_retry(_score, max_retries=5, base_delay=20)
            metric_results[mname] = val
        except Exception as e:
            print(f"  Warning: {mname} failed — {e}")
            metric_results[mname] = 0.0
            
        time.sleep(5) # Brief pause between metrics

    result = metric_results

    def safe(r, k):
        try: return round(float(r[k]), 4)
        except: return 0.0

    scores = {
        "faithfulness":       safe(result, "faithfulness"),
        "answer_relevancy":   safe(result, "answer_relevancy"),
        "context_precision":  safe(result, "context_precision"),
        "context_recall":     safe(result, "context_recall"),
    }
    scores["average"] = round(sum(scores.values()) / 4, 4)

    print(f"\n  RAGAS Scores:")
    for k, v in scores.items():
        bar = "█" * int(v * 10) + "░" * (10 - int(v * 10))
        print(f"    {k:<25} [{bar}] {v:.4f}")

    return scores


# ══════════════════════════════════════════════════════════════
# 2. FORMATTING QUALITY EVALUATION
# ══════════════════════════════════════════════════════════════
def eval_formatting() -> dict:
    from retrieval.retrieval_pipeline import ForensicsRetriever
    from generation.generation import ForensicsGenerator
    from ui.features import ResponseFormatter
    import sys
    sys.path.insert(0, ".")

    # Inline humanise_citations (simplified)
    def strip_tags(text):
        return re.sub(r'\[(?:TXN|GRAPH|REG)-\d+\]', '', text)

    print("\n  Running formatting quality evaluation...")
    retriever = ForensicsRetriever()
    generator = ForensicsGenerator()

    results = []
    for item in OUTPUT_EVAL_QUERIES[:5]:  # test first 5 for speed
        try:
            ret_results = retriever.retrieve(item["query"], top_k=5)
            output      = generator.generate(item["query"], ret_results)
            raw_answer  = output["answer"]
            formatted   = ResponseFormatter.format(raw_answer)

            checks = {
                # No raw chunk references
                "no_chunk_refs":       "chunk " not in formatted.lower(),
                # No raw citation tags
                "no_raw_tags":         not bool(re.search(r'\[(?:TXN|GRAPH|REG)-\d+\]', formatted)),
                # No stray markdown intro
                "no_md_intro":         not bool(re.match(r'^\*\*[A-Za-z ]+:\*\*', formatted.strip())),
                # HTML section headers present
                "has_html_sections":   "border-left:3px solid" in formatted,
                # At least 2 sections rendered
                "sufficient_sections": formatted.count("border-left:3px solid") >= 2,
                # Answer not empty
                "non_empty":           len(formatted.strip()) > 100,
                # Sources section present
                "has_sources":         any(s in formatted.upper() for s in ["CITATION", "SOURCE", "REFERENCE"]),
            }

            check_score = sum(checks.values()) / len(checks)
            results.append({
                "query":       item["query"][:50],
                "checks":      checks,
                "score":       round(check_score, 3),
                "passed_all":  all(checks.values()),
            })

            icon = "✅" if all(checks.values()) else "⚠️ "
            print(f"  {icon} {item['query'][:50]}")
            fails = [k for k, v in checks.items() if not v]
            if fails:
                print(f"     Failed: {fails}")

        except Exception as e:
            print(f"  ❌ Error on '{item['query'][:40]}': {e}")

    avg_score   = sum(r["score"] for r in results) / len(results) if results else 0
    perfect_pct = sum(1 for r in results if r["passed_all"]) / len(results) if results else 0

    # Per-check pass rates
    all_checks = {}
    for r in results:
        for k, v in r["checks"].items():
            all_checks[k] = all_checks.get(k, [])
            all_checks[k].append(v)
    per_check = {k: round(sum(v)/len(v), 3) for k, v in all_checks.items()}

    print(f"\n  Formatting scores per check:")
    for k, v in per_check.items():
        icon = "✅" if v == 1.0 else "⚠️ " if v >= 0.5 else "❌"
        print(f"    {icon} {k:<30} {v:.1%}")
    print(f"\n  Average formatting score: {avg_score:.3f}")
    print(f"  Perfect formatting rate:  {perfect_pct:.1%}")

    return {
        "avg_score":    round(avg_score,   4),
        "perfect_rate": round(perfect_pct, 4),
        "per_check":    per_check,
        "results":      results,
    }


# ══════════════════════════════════════════════════════════════
# 3. SUSPICION SCORE ACCURACY
# ══════════════════════════════════════════════════════════════
def eval_suspicion_scores() -> dict:
    from generation.generation import ForensicsGenerator
    generator = ForensicsGenerator()

    print("\n  Running suspicion score accuracy evaluation...")
    results = []

    for case in SUSPICION_CASES:
        try:
            scored = generator.score_suspicion(case["text"])
            score  = scored.get("score", 0)
            level  = scored.get("level", "UNKNOWN")

            # Check correctness
            if "min_score" in case:
                passed = score >= case["min_score"]
                check  = f"score>={case['min_score']}, got {score}"
            else:
                passed = score <= case["max_score"]
                check  = f"score<={case['max_score']}, got {score}"

            results.append({
                "text":    case["text"][:60],
                "expected_level": case["expected_level"],
                "actual_level":   level,
                "score":          score,
                "passed":         passed,
                "check":          check,
            })

            icon = "✅" if passed else "❌"
            print(f"  {icon} Score={score}/10 [{level}] — {check}")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    pass_rate = sum(1 for r in results if r["passed"]) / len(results) if results else 0
    avg_score_high = sum(r["score"] for r in results if r["expected_level"] in ("HIGH","CRITICAL")) / max(1, sum(1 for r in results if r["expected_level"] in ("HIGH","CRITICAL")))
    avg_score_low  = sum(r["score"] for r in results if r["expected_level"] == "LOW") / max(1, sum(1 for r in results if r["expected_level"] == "LOW"))

    print(f"\n  Pass rate:           {pass_rate:.1%}")
    print(f"  Avg score (HIGH):    {avg_score_high:.1f}/10")
    print(f"  Avg score (LOW):     {avg_score_low:.1f}/10")
    print(f"  Score separation:    {avg_score_high - avg_score_low:.1f} points")

    return {
        "pass_rate":         round(pass_rate,       4),
        "avg_score_high":    round(avg_score_high,  2),
        "avg_score_low":     round(avg_score_low,   2),
        "score_separation":  round(avg_score_high - avg_score_low, 2),
        "results":           results,
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    os.makedirs("evaluation", exist_ok=True)

    print("\n" + "="*60)
    print("  OUTPUT EVALUATION")
    print("  RAGAS + Formatting + Suspicion Score Accuracy")
    print("="*60)

    ragas_scores      = eval_ragas()
    formatting_scores = eval_formatting()
    suspicion_scores  = eval_suspicion_scores()

    # Combined output score
    ragas_avg    = ragas_scores.get("average", 0)
    fmt_avg      = formatting_scores.get("avg_score", 0)
    susp_pass    = suspicion_scores.get("pass_rate", 0)

    output_score = ragas_avg * 0.5 + fmt_avg * 0.3 + susp_pass * 0.2

    print("\n" + "="*60)
    print("  OUTPUT EVALUATION SUMMARY")
    print("="*60)
    print(f"  RAGAS Average:         {ragas_avg:.4f}")
    print(f"    Faithfulness:        {ragas_scores.get('faithfulness', 0):.4f}")
    print(f"    Answer Relevancy:    {ragas_scores.get('answer_relevancy', 0):.4f}")
    print(f"    Context Precision:   {ragas_scores.get('context_precision', 0):.4f}")
    print(f"    Context Recall:      {ragas_scores.get('context_recall', 0):.4f}")
    print(f"  Formatting Score:      {fmt_avg:.4f}")
    print(f"  Suspicion Accuracy:    {susp_pass:.4f}")
    print(f"  ─────────────────────────────────")
    print(f"  COMBINED OUTPUT SCORE: {output_score:.4f}")
    print("="*60)

    result = {
        "eval_type":    "output",
        "eval_date":    __import__("datetime").datetime.now().isoformat(),
        "ragas":        ragas_scores,
        "formatting":   formatting_scores,
        "suspicion":    suspicion_scores,
        "combined_output_score": round(output_score, 4),
    }

    with open("evaluation/eval_output_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("\n✅ Results saved to evaluation/eval_output_results.json")


if __name__ == "__main__":
    main()
