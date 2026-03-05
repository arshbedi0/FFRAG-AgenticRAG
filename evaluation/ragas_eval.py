"""
ragas_eval.py
─────────────
Evaluates the Financial Forensics RAG pipeline using RAGAS metrics.

Metrics measured:
  - Faithfulness        : Are claims grounded in retrieved context?
  - Answer Relevance    : Does the answer address the question?
  - Context Precision   : Are retrieved chunks actually relevant?
  - Context Recall      : Did we retrieve everything needed?

Also runs a BASELINE comparison (naive single-retriever, no reranker)
so you can show evaluators the improvement your hybrid pipeline delivers.

Install:
  pip install ragas datasets langchain-groq

Run from project root:
  python evaluation/ragas_eval.py
"""

import os, json, sys
sys.path.append(".")

from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
LLM_MODEL     = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
CHROMA_DIR    = os.getenv("CHROMA_DIR", "chroma_db")
RESULTS_FILE  = "evaluation/ragas_results.json"

try:
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )
except ImportError:
    raise ImportError("pip install ragas datasets")

try:
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    raise ImportError("pip install langchain-groq langchain-huggingface")

from retrieval.retrieval_pipeline import ForensicsRetriever
from generation.generation import ForensicsGenerator


# ══════════════════════════════════════════════════════════════
# GROUND TRUTH EVAL DATASET
# 25 Q&A pairs with known correct answers covering all typologies
# ══════════════════════════════════════════════════════════════
EVAL_DATASET = [
    # ── STRUCTURING ──
    {
        "question": "What is structuring and when is a SAR required for it?",
        "ground_truth": "Structuring is breaking down transactions below the $10,000 CTR threshold to evade reporting. A SAR is required when the institution knows, suspects, or has reason to suspect the transaction is designed to evade BSA reporting requirements.",
        "typology": "Structuring",
    },
    {
        "question": "Show me transactions that appear to be structuring based on amount",
        "ground_truth": "Structuring transactions have amounts just below the £9,999 reporting threshold, typically paid via cash or cheque, flagged as suspicious with Structuring typology.",
        "typology": "Structuring",
    },
    {
        "question": "What is the legal definition of structuring under BSA?",
        "ground_truth": "Under 31 C.F.R. § 1010.100(xx), structuring is defined as conducting transactions in any amount, at one or more financial institutions, on one or more days, for the purpose of evading CTR reporting requirements.",
        "typology": "Structuring",
    },

    # ── SMURFING / AGGREGATION ──
    {
        "question": "What does the smurfing network graph show?",
        "ground_truth": "The smurfing graph shows a hub-and-spoke pattern with multiple sender accounts funneling small amounts into one central hub account, consistent with placement-stage aggregation.",
        "typology": "Smurfing",
    },
    {
        "question": "What does FATF call smurfing and how is it defined?",
        "ground_truth": "FATF refers to smurfing as placement via aggregation — multiple small deposits from different sources accumulating into a single account before onward transfer, representing the placement stage of money laundering.",
        "typology": "Smurfing",
    },

    # ── LAYERING ──
    {
        "question": "Explain layering patterns in our transactions",
        "ground_truth": "Layering involves sequential transfers through multiple accounts to obscure the audit trail. The layering graph shows 67 accounts with £1.7M suspicious volume, using ACH transfers of £20,000-£75,000 moving through chains of accounts.",
        "typology": "Layering",
    },
    {
        "question": "What does the professional money laundering report say about layering?",
        "ground_truth": "The FATF Professional Money Laundering report describes layering as the use of complex schemes including shell company accounts and proxy structures to move illicit proceeds, managed by individuals coordinating financial transactions.",
        "typology": "Layering",
    },

    # ── HIGH RISK CORRIDOR ──
    {
        "question": "Which accounts are sending money to UAE and why is it suspicious?",
        "ground_truth": "Account 176667861 sent multiple cross-border transfers to UAE totaling over £86,000, flagged as High_Risk_Corridor typology. UK to UAE flows are suspicious due to currency mismatches and FATF high-risk jurisdiction indicators.",
        "typology": "High_Risk_Corridor",
    },
    {
        "question": "What FATF recommendation applies to high risk geographic corridors?",
        "ground_truth": "FATF Recommendation 19 requires enhanced due diligence for transactions from or to high-risk jurisdictions. Cross-border transfers to UAE, Turkey, and Morocco trigger enhanced monitoring obligations.",
        "typology": "High_Risk_Corridor",
    },
    {
        "question": "Show me all suspicious cross-border transactions to high risk countries",
        "ground_truth": "Transactions flagged as High_Risk_Corridor involve UK accounts sending cross-border payments to UAE, Turkey, Morocco, and Nigeria, with amounts ranging from £17,000 to £46,000, received in local currencies.",
        "typology": "High_Risk_Corridor",
    },

    # ── DORMANT REACTIVATION ──
    {
        "question": "Find dormant accounts that were suddenly reactivated with large amounts",
        "ground_truth": "Dormant reactivation transactions show accounts suddenly processing £60,000-£82,000 cash transfers after inactivity. The network graph shows £2.2M total volume with 98% suspicious, involving satellite nodes with single thick edges.",
        "typology": "Dormant_Reactivation",
    },
    {
        "question": "What is the risk of dormant account reactivation?",
        "ground_truth": "Dormant account reactivation is a HIGH risk placement indicator. Accounts kept inactive to avoid monitoring thresholds are suddenly activated for large single transfers, consistent with initial placement of criminal proceeds.",
        "typology": "Dormant_Reactivation",
    },

    # ── SAR FILING ──
    {
        "question": "What is the SAR filing timeline for continuing suspicious activity?",
        "ground_truth": "According to FinCEN guidance: Day 0 is detection, Day 30 is initial SAR filing, Day 120 is end of 90-day period, Day 150 is SAR for continued activity. The filing deadline for continuing activity is 120 calendar days after the previous SAR.",
        "typology": "SAR_Compliance",
    },
    {
        "question": "Is a financial institution required to document the decision not to file a SAR?",
        "ground_truth": "No. There is no requirement under the BSA or its implementing regulations for a financial institution to document its decision not to file a SAR, though FinCEN has previously encouraged but not required such documentation.",
        "typology": "SAR_Compliance",
    },
    {
        "question": "How many days does a bank have to file a SAR after detecting suspicious activity?",
        "ground_truth": "A SAR must be filed no later than 30 calendar days from the date of initial detection. An additional 30 calendar days is allowed if no suspect is identified, but reporting must not be delayed more than 60 days.",
        "typology": "SAR_Compliance",
    },

    # ── CURRENCY MISMATCH ──
    {
        "question": "What are currency mismatch transactions and why are they suspicious?",
        "ground_truth": "Currency mismatch transactions occur when the received currency is inconsistent with the receiver bank location, suggesting trade-based money laundering or deliberate currency obfuscation to obscure fund flows.",
        "typology": "Currency_Mismatch",
    },
    {
        "question": "Show transactions where payment currency does not match receiver location",
        "ground_truth": "Currency mismatch transactions show accounts in UAE receiving UK pounds, or Turkey accounts receiving Euros instead of Turkish lira, flagged as suspicious with Currency_Mismatch typology.",
        "typology": "Currency_Mismatch",
    },

    # ── ROUND TRIP ──
    {
        "question": "What is a round trip transaction in AML terms?",
        "ground_truth": "Round trip transactions involve funds leaving an account, passing through intermediaries in different jurisdictions, and returning to a beneficial owner-connected account. This represents integration-stage laundering to make criminal funds appear as legitimate returns.",
        "typology": "Round_Trip",
    },
    {
        "question": "Show the round trip transaction network graph analysis",
        "ground_truth": "The round trip graph shows circular subgraphs where funds complete 3-hop circuits returning to originating country accounts after transiting through UAE and Turkish intermediaries, with suspicious volume of £1.35M.",
        "typology": "Round_Trip",
    },

    # ── RAPID SUCCESSION ──
    {
        "question": "What is rapid succession transaction pattern?",
        "ground_truth": "Rapid succession involves high-frequency transactions between a small cluster of accounts in a tight time window, with deliberately varied amounts to avoid pattern detection, used to generate artificial volume or overwhelm monitoring systems.",
        "typology": "Rapid_Succession",
    },

    # ── GENERAL AML ──
    {
        "question": "What are the three stages of money laundering according to FATF?",
        "ground_truth": "The three stages are placement (introducing criminal proceeds into the financial system), layering (obscuring the audit trail through complex transactions), and integration (making funds appear legitimate through reinvestment).",
        "typology": "General_AML",
    },
    {
        "question": "What is the FATF and what do its recommendations cover?",
        "ground_truth": "The FATF is an independent inter-governmental body that develops policies to protect the global financial system against money laundering, terrorist financing, and proliferation financing. Its 40 Recommendations are the global AML/CFT standard.",
        "typology": "General_AML",
    },
    {
        "question": "What transactions involve accounts in Turkey?",
        "ground_truth": "Transactions involving Turkey include High_Risk_Corridor flows from UK accounts to Turkish receivers, and Currency_Mismatch cases where Turkish lira is received despite different sending location, all flagged as suspicious.",
        "typology": "High_Risk_Corridor",
    },
    {
        "question": "What is the total suspicious volume in the layering network graph?",
        "ground_truth": "The layering network graph shows a total suspicious volume of £1,700,344.35 across 45 transactions involving 67 accounts, making it the highest suspicious volume typology in the dataset.",
        "typology": "Layering",
    },
    {
        "question": "What payment types are most associated with suspicious transactions?",
        "ground_truth": "Cash and cheque payments dominate structuring cases. Cross-border payments are associated with High_Risk_Corridor typology. ACH transfers are the primary payment type in layering transactions.",
        "typology": "General_AML",
    },
]


# ══════════════════════════════════════════════════════════════
# BASELINE RETRIEVER (naive — dense only, no reranker)
# For before/after comparison
# ══════════════════════════════════════════════════════════════
class BaselineRetriever:
    """Simple dense-only retrieval from a single collection — no BM25, no reranker."""

    def __init__(self):
        import chromadb
        from chromadb.utils import embedding_functions
        EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self.collections = {
            "transactions":   client.get_collection("transactions",   embedding_function=ef),
            "graph_captions": client.get_collection("graph_captions", embedding_function=ef),
            "regulations":    client.get_collection("regulations",    embedding_function=ef),
        }

    def retrieve(self, query: str, top_k: int = 5) -> dict:
        all_docs = []
        results  = {"transactions": [], "graph_captions": [], "regulations": []}

        for col_name, col in self.collections.items():
            r = col.query(
                query_texts=[query],
                n_results=min(top_k, col.count()),
                include=["documents", "metadatas"]
            )
            for i in range(len(r["ids"][0])):
                doc = {
                    "id":         r["ids"][0][i],
                    "document":   r["documents"][0][i],
                    "metadata":   r["metadatas"][0][i],
                    "collection": col_name,
                }
                all_docs.append(doc)
                results[col_name].append(doc)

        results["all_results"] = all_docs[:top_k]
        results["rewritten"]   = [query]
        results["query"]       = query
        return results


# ══════════════════════════════════════════════════════════════
# RAGAS EVALUATOR
# ══════════════════════════════════════════════════════════════
class RAGASEvaluator:

    def __init__(self):
        self.llm = ChatGroq(
            model=LLM_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.0,
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        )
        self.metrics = [
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ]

    def build_samples(
        self,
        eval_data: list[dict],
        retriever,
        generator: ForensicsGenerator,
        label: str = "pipeline",
    ) -> list[SingleTurnSample]:
        samples = []
        total   = len(eval_data)

        print(f"\n  Building {label} samples ({total} queries)...")

        for i, item in enumerate(eval_data, 1):
            question     = item["question"]
            ground_truth = item["ground_truth"]

            print(f"  [{i:02d}/{total}] {question[:60]}...")

            try:
                # Retrieve
                results = retriever.retrieve(question, top_k=5)

                # Collect all context strings
                contexts = []
                for r in results.get("all_results", []):
                    contexts.append(r["document"])

                if not contexts:
                    contexts = ["No relevant context found."]

                # Generate answer
                output = generator.generate(question, results)
                answer = output["answer"]

                samples.append(SingleTurnSample(
                    user_input=question,
                    response=answer,
                    retrieved_contexts=contexts,
                    reference=ground_truth,
                ))

            except Exception as e:
                print(f"     ⚠️  Error on query {i}: {e}")
                samples.append(SingleTurnSample(
                    user_input=question,
                    response="Error generating response.",
                    retrieved_contexts=["Error retrieving context."],
                    reference=ground_truth,
                ))

        return samples

    def evaluate(self, samples: list[SingleTurnSample]) -> dict:
        dataset = EvaluationDataset(samples=samples)
        result  = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings,
        )
        return result


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    import time
    os.makedirs("evaluation", exist_ok=True)

    print(f"\n{'='*65}")
    print("  FINANCIAL FORENSICS RAG — RAGAS EVALUATION")
    print(f"{'='*65}")
    print(f"  Queries    : {len(EVAL_DATASET)}")
    print(f"  Metrics    : Faithfulness, Answer Relevancy,")
    print(f"               Context Precision, Context Recall")
    print(f"  Comparison : Hybrid Pipeline vs Naive Baseline")
    print(f"{'='*65}\n")

    # ── Load pipelines ──
    print("Loading pipelines...")
    hybrid_retriever   = ForensicsRetriever()
    baseline_retriever = BaselineRetriever()
    generator          = ForensicsGenerator()
    evaluator          = RAGASEvaluator()

    # ── HYBRID PIPELINE EVAL ──
    print(f"\n{'─'*65}")
    print("  PHASE 1 — Hybrid Pipeline (BM25 + Dense + Reranker)")
    print(f"{'─'*65}")
    t0 = time.time()
    hybrid_samples = evaluator.build_samples(
        EVAL_DATASET, hybrid_retriever, generator, label="Hybrid"
    )
    hybrid_results = evaluator.evaluate(hybrid_samples)
    hybrid_time    = time.time() - t0

    # ── BASELINE EVAL ──
    print(f"\n{'─'*65}")
    print("  PHASE 2 — Baseline (Dense Only, No Reranker)")
    print(f"{'─'*65}")
    t0 = time.time()
    baseline_samples = evaluator.build_samples(
        EVAL_DATASET, baseline_retriever, generator, label="Baseline"
    )
    baseline_results = evaluator.evaluate(baseline_samples)
    baseline_time    = time.time() - t0

    # ── RESULTS ──
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    def safe_score(result, metric):
        try:
            return float(result[metric])
        except Exception:
            return 0.0

    hybrid_scores   = {m: safe_score(hybrid_results,   m) for m in metrics}
    baseline_scores = {m: safe_score(baseline_results, m) for m in metrics}

    print(f"\n{'='*65}")
    print("  EVALUATION RESULTS")
    print(f"{'='*65}")
    print(f"\n  {'Metric':<25} {'Baseline':>10} {'Hybrid':>10} {'Delta':>10}")
    print(f"  {'─'*55}")

    for m in metrics:
        b = baseline_scores[m]
        h = hybrid_scores[m]
        d = h - b
        arrow = "↑" if d > 0 else "↓" if d < 0 else "="
        print(f"  {m:<25} {b:>10.4f} {h:>10.4f} {arrow}{abs(d):>8.4f}")

    avg_hybrid   = sum(hybrid_scores.values())   / len(metrics)
    avg_baseline = sum(baseline_scores.values()) / len(metrics)
    avg_delta    = avg_hybrid - avg_baseline

    print(f"  {'─'*55}")
    print(f"  {'AVERAGE':<25} {avg_baseline:>10.4f} {avg_hybrid:>10.4f} "
          f"{'↑' if avg_delta>0 else '↓'}{abs(avg_delta):>8.4f}")
    print(f"\n  Hybrid eval time   : {hybrid_time:.0f}s")
    print(f"  Baseline eval time : {baseline_time:.0f}s")

    # ── PER-TYPOLOGY BREAKDOWN ──
    print(f"\n{'─'*65}")
    print("  PER-TYPOLOGY BREAKDOWN (Hybrid Pipeline)")
    print(f"{'─'*65}")

    typology_groups = {}
    for item, sample in zip(EVAL_DATASET, hybrid_samples):
        t = item["typology"]
        if t not in typology_groups:
            typology_groups[t] = []
        typology_groups[t].append(sample)

    for typology, t_samples in sorted(typology_groups.items()):
        if not t_samples:
            continue
        t_results = evaluator.evaluate(t_samples)
        t_scores  = {m: safe_score(t_results, m) for m in metrics}
        t_avg     = sum(t_scores.values()) / len(metrics)
        status    = "✅" if t_avg >= 0.7 else "⚠️ " if t_avg >= 0.5 else "❌"
        print(f"  {status} {typology:<28} avg={t_avg:.3f} | "
              f"F={t_scores['faithfulness']:.2f} "
              f"AR={t_scores['answer_relevancy']:.2f} "
              f"CP={t_scores['context_precision']:.2f} "
              f"CR={t_scores['context_recall']:.2f}")

    # ── SAVE RESULTS ──
    output = {
        "eval_date":       __import__("datetime").datetime.now().isoformat(),
        "n_queries":       len(EVAL_DATASET),
        "metrics":         metrics,
        "hybrid": {
            "scores":      hybrid_scores,
            "average":     avg_hybrid,
            "eval_time_s": hybrid_time,
        },
        "baseline": {
            "scores":      baseline_scores,
            "average":     avg_baseline,
            "eval_time_s": baseline_time,
        },
        "improvement": {
            m: hybrid_scores[m] - baseline_scores[m] for m in metrics
        },
        "average_improvement": avg_delta,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Results saved to {RESULTS_FILE}")
    print(f"\n{'='*65}")
    print(f"  SUMMARY FOR EVALUATORS")
    print(f"{'='*65}")
    print(f"  Hybrid pipeline scored {avg_hybrid:.3f} average across 4 RAGAS metrics")
    print(f"  vs baseline naive retrieval at {avg_baseline:.3f}")
    print(f"  → {abs(avg_delta)*100:.1f}% {'improvement' if avg_delta > 0 else 'difference'} "
          f"from hybrid BM25 + dense + reranker architecture")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
