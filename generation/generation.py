"""
generation.py
─────────────
Groq LLM generation layer for the Financial Forensics RAG.

Takes retrieved context (transactions + graphs + regulations)
and generates a structured forensic answer with citations.

Usage:
  from generation.generation import ForensicsGenerator
  gen = ForensicsGenerator()
  answer = gen.generate(query, retrieval_results)

Or run standalone to test:
  python generation/generation.py
"""


import os

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

GROQ_API_KEY  = get_config("GROQ_API_KEY")
LLM_MODEL     = get_config("LLM_MODEL", "llama-3.3-70b-versatile")
MAX_TOKENS    = get_config("MAX_TOKENS", 1024, int)
TEMPERATURE   = get_config("TEMPERATURE", 0.1, float)  # low = factual, consistent

try:
    from groq import Groq
except ImportError:
    raise ImportError("pip install groq")


# ══════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are FFRAG — a Financial Forensics AI assistant specialising in \
Anti-Money Laundering (AML) investigation. You help financial analysts investigate \
suspicious transactions by reasoning across transaction records, wallet network graphs, \
and regulatory documents simultaneously.

YOUR BEHAVIOUR:
- Always ground your answers in the provided context. Never hallucinate transaction details.
- - Cite your sources naturally in prose. Instead of "[TXN-1]", write 
  "Account 176667861 (Transaction Record)". Instead of "[REG-1]", write 
  "FinCEN bulletin-2025-31a (SAR FAQ)". Instead of "[GRAPH-1]", write 
  "Smurfing Network Graph Analysis". Always reference the actual account 
  number, document name, or graph type — never raw bracket tags.
- If the context doesn't contain enough information, say so clearly.
- When transactions are involved, always state: account numbers, amounts, locations, typology.
- When regulations are involved, always cite the specific document and section.
- Be precise and structured — analysts rely on your output for SAR filings.

AML TERMINOLOGY MAPPING (FATF uses formal terms, not street slang):
- "Smurfing"       → FATF calls this "placement via aggregation" or "structuring"
- "Money mules"    → FATF calls these "third-party placement agents"
- "Layering"       → consistent across FATF and FinCEN
- "Round tripping" → FATF calls this "carousel transactions" or "loan-back schemes"
- "Shell company"  → FATF calls these "legal persons used as vehicles"
When a query uses informal terms and the regulation uses formal ones, bridge the gap
in your answer — explain both the informal and formal terminology explicitly.

OUTPUT FORMAT (always follow this structure):
1. FINDINGS       — what the evidence shows
2. TYPOLOGY MATCH — which AML pattern this matches and why  
3. REGULATORY     — which regulation/recommendation applies
4. RISK VERDICT   — suspicion score (1-10) + one-line SAR summary
5. SOURCES USED   — list which [TXN], [GRAPH], [REG] citations you used

If asked a general question (not about specific transactions), skip sections 1-2 and answer directly."""


# ══════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════════════════════
def build_prompt(query: str, context: str) -> str:
    return f"""RETRIEVED CONTEXT:
{context}

─────────────────────────────────────────
ANALYST QUERY: {query}
─────────────────────────────────────────

Based strictly on the context above, provide your forensic analysis."""


# ══════════════════════════════════════════════════════════════
# GENERATOR CLASS
# ══════════════════════════════════════════════════════════════
class ForensicsGenerator:

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not set in .env\n"
                "Get a free key at: https://console.groq.com"
            )
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model  = LLM_MODEL
        print(f"✅ ForensicsGenerator ready — model: {self.model}")

    def generate(
        self,
        query: str,
        retrieval_results: dict,
        context_str: str = None,
    ) -> dict:
        """
        Generate a forensic answer from retrieval results.

        Args:
            query:            User's original question
            retrieval_results: Output dict from ForensicsRetriever.retrieve()
            context_str:      Pre-formatted context string (optional).
                              If None, formats from retrieval_results automatically.

        Returns:
            {
                "query":    original query,
                "answer":   LLM forensic answer,
                "sources":  which collections were used,
                "model":    model used,
                "usage":    token counts,
            }
        """
        # Format context if not pre-formatted
        if context_str is None:
            context_str = self._format_context(retrieval_results)

        if not context_str.strip():
            return {
                "query":   query,
                "answer":  "No relevant context found in the database for this query.",
                "sources": [],
                "model":   self.model,
                "usage":   {},
            }

        # Build prompt
        prompt = build_prompt(query, context_str)

        # Call Groq
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        answer = response.choices[0].message.content.strip()
        usage  = {
            "prompt_tokens":     response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens":      response.usage.total_tokens,
        }

        # Track which source types were used
        sources = []
        if retrieval_results.get("transactions"):
            sources.append("transactions")
        if retrieval_results.get("graph_captions"):
            sources.append("graph_captions")
        if retrieval_results.get("regulations"):
            sources.append("regulations")

        return {
            "query":   query,
            "answer":  answer,
            "sources": sources,
            "model":   self.model,
            "usage":   usage,
        }

    # ──────────────────────────────────────────────
    # CONTEXT FORMATTER
    # ──────────────────────────────────────────────
    def _format_context(self, results: dict) -> str:
        """Formats retrieval results into LLM-readable context."""
        sections = []

        if results.get("transactions"):
            sections.append("=== TRANSACTION RECORDS ===")
            for i, r in enumerate(results["transactions"], 1):
                m = r["metadata"]
                sections.append(
                    f"[TXN-{i}] {r['document']}\n"
                    f"   Typology: {m.get('typology','?')} | "
                    f"Suspicious: {'YES' if m.get('is_suspicious') else 'NO'} | "
                    f"Amount: £{float(m.get('amount', 0)):,.2f}"
                )

        if results.get("graph_captions"):
            sections.append("\n=== WALLET NETWORK GRAPH ANALYSIS ===")
            for i, r in enumerate(results["graph_captions"], 1):
                sections.append(
                    f"[GRAPH-{i}] {r['document'][:800]}\n"
                    f"   Graph ID: {r['metadata'].get('graph_id','?')}"
                )

        if results.get("regulations"):
            sections.append("\n=== REGULATORY REFERENCES ===")
            for i, r in enumerate(results["regulations"], 1):
                sections.append(
                    f"[REG-{i}] Source: {r['metadata'].get('filename','?')} "
                    f"(chunk {r['metadata'].get('chunk_idx','?')})\n"
                    f"{r['document']}"
                )

        return "\n\n".join(sections)

    # ──────────────────────────────────────────────
    # SUSPICION SCORER (bonus utility)
    # ──────────────────────────────────────────────
    def score_suspicion(self, transaction_text: str) -> dict:
        """
        Standalone method: give it a transaction description,
        returns a suspicion score 1-10 with reasoning.
        Useful for the UI badge display.
        """
        prompt = f"""Rate the following transaction's suspicion level from 1-10.

Transaction: {transaction_text}

Respond in this exact JSON format:
{{
  "score": <1-10>,
  "level": "<LOW|MEDIUM|HIGH|CRITICAL>",
  "reason": "<one sentence explaining the score>",
  "flags": ["<flag1>", "<flag2>"]
}}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an AML risk scoring expert. Respond only with valid JSON."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=256,
            temperature=0.0,
        )

        import json
        try:
            return json.loads(response.choices[0].message.content.strip())
        except json.JSONDecodeError:
            return {"score": 0, "level": "UNKNOWN", "reason": "Parse error", "flags": []}


# ══════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    sys.path.append(".")

    from retrieval.retrieval_pipeline import ForensicsRetriever

    retriever = ForensicsRetriever()
    generator = ForensicsGenerator()

    # ── TEST QUERIES ──
    TEST_QUERIES = [
        "Which accounts are sending money to UAE and why is that suspicious?",
        "Explain the SAR filing timeline I should follow for a continuing structuring case",
        "Find dormant accounts that were suddenly reactivated — what does the graph show?",
        "What does FATF say about smurfing and which transactions in our data match?",
    ]

    for query in TEST_QUERIES:
        print(f"\n{'═'*65}")
        print(f"  QUERY: {query}")
        print(f"{'═'*65}")

        # Retrieve
        results = retriever.retrieve(query, top_k=5)

        # Generate
        output = generator.generate(query, results)

        print(f"\n🤖 FORENSIC ANSWER [{output['model']}]")
        print(f"   Sources used: {output['sources']}")
        print(f"   Tokens: {output['usage']}")
        print(f"\n{output['answer']}")

        # Bonus: score suspicion on first transaction if any
        if results["transactions"]:
            first_txn = results["transactions"][0]["document"]
            score = generator.score_suspicion(first_txn)
            print(f"\n🚨 SUSPICION SCORE: {score['score']}/10 [{score['level']}]")
            print(f"   Reason: {score['reason']}")
            print(f"   Flags:  {score['flags']}")

        print()
