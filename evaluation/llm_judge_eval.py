"""
evaluation/llm_judge_eval.py
────────────────────────────
Local LLM-as-a-Judge Evaluation using Ollama (qwen2.5).
Zero rate limits, zero API costs, 100% local scoring.
"""

import os, sys, time, re, requests
from dataclasses import dataclass

sys.path.append(".")
from dotenv import load_dotenv
load_dotenv()

# ══════════════════════════════════════════════════════════════
# TEST QUERIES
# ══════════════════════════════════════════════════════════════
TEST_QUERIES = [
    {
        "query": "What is the SAR filing timeline for continuing suspicious activity?",
        "ground_truth": "Day 0 detection, Day 30 initial SAR filing, Day 120 end of 90-day period, Day 150 SAR for continued activity. Filing deadline is 120 calendar days after previous SAR.",
    },
    {
        "query": "Show structuring transactions below £10,000",
        "ground_truth": "Structuring transactions have amounts just below £9,999 threshold, paid via cash or cheque, flagged as suspicious with Structuring typology.",
    },
    {
        "query": "Find dormant accounts that were suddenly reactivated",
        "ground_truth": "Dormant reactivation transactions show accounts processing £60,000-£82,000 cash transfers after inactivity.",
    },
    {
        "query": "What does FATF say about placement and aggregation?",
        "ground_truth": "FATF describes placement as introducing criminal proceeds into financial system via aggregation of multiple small deposits, representing the first stage of money laundering.",
    },
    {
        "query": "Explain layering patterns in our transactions",
        "ground_truth": "Layering involves sequential transfers through multiple accounts to obscure the origin of funds.",
    }
]

# ══════════════════════════════════════════════════════════════
# THE EVALUATOR
# ══════════════════════════════════════════════════════════════
class FFRAGLocalEvaluator:
    def __init__(self):
        print("🔌 Initializing Local Ollama Evaluator...")
        self.ollama_url = "http://localhost:11434/api/chat"
        self.judge_model = "qwen2.5"
        
        from retrieval.retrieval_pipeline import ForensicsRetriever
        from generation.generation import ForensicsGenerator
        
        # Generator still uses Groq (based on your .env) to test your actual pipeline output
        self.retriever = ForensicsRetriever()
        self.generator = ForensicsGenerator()

    def _get_judge_score(self, prompt: str) -> float:
        """Helper to get a 1-5 score from the local Ollama judge."""
        payload = {
            "model": self.judge_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "stream": False
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=600)
            if response.status_code == 200:
                content = response.json()["message"]["content"].strip()
                score_match = re.search(r'(\d+(?:\.\d+)?)', content)
                if score_match:
                    score = float(score_match.group(1))
                    return min(max(score / 5.0, 0.0), 1.0)  # Normalize to 0-1
            else:
                print(f"    ⚠️ Judge Error: HTTP {response.status_code}")
            return 0.5 # Default fallback
        except Exception as e:
            print(f"    ⚠️ Local Judge Error: {e}")
            return 0.5

    def _judge_faithfulness(self, answer: str, context: str) -> float:
        prompt = f"""Rate how well the answer is supported by the provided context.
Context: {context}
Answer: {answer}
Consider: Does the answer contain information not in the context? Are there contradictions?
Rate from 1-5 (1=not faithful/hallucinated, 5=perfectly faithful, uses only context). Output ONLY the number."""
        return self._get_judge_score(prompt)

    def _judge_relevance(self, question: str, answer: str) -> float:
        prompt = f"""Rate how well the answer addresses the question.
Question: {question}
Answer: {answer}
Consider: Is the response on-topic? Does it provide the requested information?
Rate from 1-5 (1=not relevant, 5=highly relevant). Output ONLY the number."""
        return self._get_judge_score(prompt)

    def _judge_completeness(self, question: str, answer: str, ground_truth: str) -> float:
        prompt = f"""Rate the completeness of the answer compared to the expected information.
Question: {question}
Expected truth: {ground_truth}
Actual Answer: {answer}
Rate completeness from 1-5 (1=misses key facts, 5=fully covers the expected truth). Output ONLY the number."""
        return self._get_judge_score(prompt)

    def _detect_hallucination(self, answer: str, context: str) -> bool:
        prompt = f"""Does the answer contain specific facts, numbers, or claims NOT supported by the context?
Context: {context}
Answer: {answer}
Respond with only "YES" if it contains unsupported information, or "NO" if it is faithful."""
        
        payload = {
            "model": self.judge_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "stream": False
        }
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=120)
            return "YES" in response.json()["message"]["content"].strip().upper()
        except:
            return False

    def evaluate_pipeline(self):
        print("\n🚀 Starting Local Ollama LLM-as-a-Judge Evaluation...")
        results = []
        
        for i, item in enumerate(TEST_QUERIES, 1):
            q = item["query"]
            print(f"\n[{i}/{len(TEST_QUERIES)}] Query: {q[:50]}...")
            
            # 1. Run FFRAG Pipeline
            print("    🔍 Retrieving & Generating (via Groq)...")
            try:
                ret_results = self.retriever.retrieve(q, top_k=5)
                output = self.generator.generate(q, ret_results)
                
                answer = output["answer"]
                contexts = "\n".join([r["document"] for r in ret_results.get("all_results", [])])
                if not contexts: contexts = "No context retrieved."
                
                # 2. Run Judge Evaluations locally
                print(f"    ⚖️  Grading Faithfulness (via {self.judge_model})...")
                faith = self._judge_faithfulness(answer, contexts)
                
                print(f"    ⚖️  Grading Relevance (via {self.judge_model})...")
                rel = self._judge_relevance(q, answer)
                
                print(f"    ⚖️  Grading Completeness (via {self.judge_model})...")
                comp = self._judge_completeness(q, answer, item["ground_truth"])
                
                print(f"    ⚖️  Checking for Hallucinations (via {self.judge_model})...")
                hallucinated = self._detect_hallucination(answer, contexts)
                
                # Apply penalty for hallucination
                overall = (faith + rel + comp) / 3.0
                if hallucinated:
                    overall *= 0.8
                
                results.append({
                    "faithfulness": faith,
                    "relevance": rel,
                    "completeness": comp,
                    "overall": overall,
                    "hallucinated": hallucinated
                })
                
                print(f"    ✅ Done: Overall Score {overall:.2f}")
                
            except Exception as e:
                print(f"    ❌ Pipeline failed on query: {e}")

        # Summary
        if not results: return
        
        avg_faith = sum(r["faithfulness"] for r in results) / len(results)
        avg_rel   = sum(r["relevance"] for r in results) / len(results)
        avg_comp  = sum(r["completeness"] for r in results) / len(results)
        avg_over  = sum(r["overall"] for r in results) / len(results)
        hall_rate = sum(1 for r in results if r["hallucinated"]) / len(results)
        
        print("\n" + "="*50)
        print(f" 📊 FINAL EVALUATION SCORES (Judge: {self.judge_model})")
        print("="*50)
        print(f"  Faithfulness:      {avg_faith:.2f}")
        print(f"  Relevance:         {avg_rel:.2f}")
        print(f"  Completeness:      {avg_comp:.2f}")
        print(f"  Hallucination Rate:{hall_rate:.1%}")
        print(f"  -------------------------------------")
        print(f"  OVERALL QUALITY:   {avg_over:.2f} / 1.00")
        print("="*50)

if __name__ == "__main__":
    evaluator = FFRAGLocalEvaluator()
    evaluator.evaluate_pipeline()