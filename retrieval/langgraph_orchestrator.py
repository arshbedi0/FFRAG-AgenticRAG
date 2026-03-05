"""
langgraph_orchestrator.py
──────────────────────────
Agentic RAG control loop using LangGraph.

Nodes:
  A. Router          — classifies query → pipeline selection
  B. QueryRewriter   — expands query + HyDE
  C. Retriever       — hybrid retrieval + context optimization
  D. Grader          — self-reflection, loops back if relevance low
  E. Generator       — final SAR-style answer

State flows:
  query → Router → QueryRewriter → Retriever → Grader → Generator
                                      ↑              |
                                      └──────────────┘ (if relevance low)

Install:
  pip install langgraph langchain-groq

Run:
  python retrieval/langgraph_orchestrator.py
"""

import os, json
from typing import TypedDict, Literal, Annotated
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL    = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
MAX_RETRIES  = 2   # max rewrite+retrieve loops before accepting

try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
except ImportError:
    raise ImportError("pip install langgraph")

from groq import Groq


# ══════════════════════════════════════════════════════════════
# STATE DEFINITION
# ══════════════════════════════════════════════════════════════
class AgentState(TypedDict):
    # Input
    query:            str

    # Router output
    pipeline:         str           # "vector" | "graph" | "both"
    query_intent:     str           # "numerical" | "conceptual" | "compliance"

    # Rewriter output
    rewritten_queries: list[str]
    hyde_document:     str          # hypothetical document for HyDE

    # Retrieval output
    raw_results:       dict         # from ForensicsRetriever
    optimized_context: str          # after context_optimizer
    all_chunks:        list[dict]

    # Grader output
    relevance_score:   float        # 0.0 – 1.0
    grader_feedback:   str          # why it scored low
    retry_count:       int

    # Generator output
    answer:            str
    sources:           list[str]
    suspicion_score:   dict | None

    # Control
    should_retry:      bool


# ══════════════════════════════════════════════════════════════
# GROQ CLIENT (shared)
# ══════════════════════════════════════════════════════════════
groq_client = Groq(api_key=GROQ_API_KEY)

def llm(prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    """Lightweight LLM call via Groq."""
    r = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return r.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════════════
# NODE A: ROUTER
# ══════════════════════════════════════════════════════════════
ROUTER_PROMPT = """You are an AML query router. Classify the user query into:

1. PIPELINE:
   - "vector"  : conceptual questions, typology explanations, regulatory lookups
   - "graph"   : specific account tracing, multi-hop fund flows, numerical aggregation
   - "both"    : queries requiring both conceptual and structural analysis

2. INTENT:
   - "numerical"   : aggregations, account totals, hop counts, specific amounts
   - "conceptual"  : typology definitions, regulatory explanations, pattern analysis
   - "compliance"  : SAR filing, regulatory requirements, FATF recommendations

Respond ONLY with valid JSON: {{"pipeline": "...", "intent": "..."}}

QUERY: {query}"""

def router_node(state: AgentState) -> AgentState:
    """Classify query and decide which pipeline to use."""
    print(f"\n  [ROUTER] Query: {state['query'][:60]}...")

    try:
        response = llm(ROUTER_PROMPT.format(query=state["query"]), max_tokens=100)
        # Strip markdown fences if present
        clean = response.replace("```json","").replace("```","").strip()
        parsed = json.loads(clean)
        pipeline = parsed.get("pipeline", "both")
        intent   = parsed.get("intent",   "conceptual")
    except Exception:
        pipeline = "both"
        intent   = "conceptual"

    print(f"  [ROUTER] Pipeline={pipeline}, Intent={intent}")

    return {
        **state,
        "pipeline":     pipeline,
        "query_intent": intent,
        "retry_count":  state.get("retry_count", 0),
    }


# ══════════════════════════════════════════════════════════════
# NODE B: QUERY REWRITER (with HyDE)
# ══════════════════════════════════════════════════════════════
REWRITER_PROMPT = """You are an AML forensics query expansion specialist.

Given the user's query, generate 3 alternative search queries that:
1. Use different but semantically equivalent phrasing
2. Include relevant AML/FATF terminology
3. Cover different aspects of the same question

Also generate a HYPOTHETICAL DOCUMENT — a short paragraph that the perfect
answer document would contain. This is used for HyDE (Hypothetical Document Embedding).

Respond ONLY with valid JSON:
{{
  "queries": ["query1", "query2", "query3"],
  "hyde_document": "A short paragraph the ideal source document would contain..."
}}

ORIGINAL QUERY: {query}
INTENT: {intent}
PRIOR FEEDBACK (if any): {feedback}"""

def query_rewriter_node(state: AgentState) -> AgentState:
    """Expand query into variants + generate HyDE document."""
    feedback = state.get("grader_feedback", "none")
    print(f"  [REWRITER] Expanding query (retry={state.get('retry_count',0)})...")

    try:
        response = llm(
            REWRITER_PROMPT.format(
                query=state["query"],
                intent=state.get("query_intent", "conceptual"),
                feedback=feedback,
            ),
            max_tokens=400,
            temperature=0.3,
        )
        clean  = response.replace("```json","").replace("```","").strip()
        parsed = json.loads(clean)
        queries = [state["query"]] + parsed.get("queries", [])[:3]
        hyde    = parsed.get("hyde_document", "")
    except Exception:
        queries = [state["query"]]
        hyde    = ""

    print(f"  [REWRITER] Generated {len(queries)} query variants")
    if hyde:
        print(f"  [REWRITER] HyDE: {hyde[:80]}...")

    return {**state, "rewritten_queries": queries, "hyde_document": hyde}


# ══════════════════════════════════════════════════════════════
# NODE C: RETRIEVER + CONTEXT OPTIMIZER
# ══════════════════════════════════════════════════════════════
def retriever_node(state: AgentState) -> AgentState:
    """
    Run hybrid retrieval across all query variants,
    then optimize context (compression + reordering).
    """
    import sys
    sys.path.append(".")
    from retrieval.retrieval_pipeline import ForensicsRetriever
    from retrieval.context_optimizer  import ContextOptimizer

    print(f"  [RETRIEVER] Running retrieval for {len(state['rewritten_queries'])} queries...")

    retriever  = ForensicsRetriever()
    optimizer  = ContextOptimizer(use_compression=False)  # compression optional

    # Merge results from all query variants
    merged_results = {
        "transactions":   [],
        "graph_captions": [],
        "regulations":    [],
        "all_results":    [],
    }

    seen_ids = set()
    for q in state["rewritten_queries"]:
        results = retriever.retrieve(q, top_k=5)
        for col in ["transactions", "graph_captions", "regulations"]:
            for r in results.get(col, []):
                rid = r.get("id", r.get("document", "")[:50])
                if rid not in seen_ids:
                    merged_results[col].append(r)
                    merged_results["all_results"].append({**r, "collection": col})
                    seen_ids.add(rid)

    # If HyDE document provided, run one extra retrieval pass with it
    if state.get("hyde_document"):
        print(f"  [RETRIEVER] HyDE retrieval pass...")
        hyde_results = retriever.retrieve(state["hyde_document"], top_k=3)
        for col in ["transactions", "graph_captions", "regulations"]:
            for r in hyde_results.get(col, []):
                rid = r.get("id", r.get("document","")[:50])
                if rid not in seen_ids:
                    merged_results[col].append(r)
                    merged_results["all_results"].append({**r, "collection": col})
                    seen_ids.add(rid)

    all_chunks = merged_results["all_results"]
    print(f"  [RETRIEVER] Total unique chunks: {len(all_chunks)}")

    # Context optimization
    opt = optimizer.optimize(
        query=state["query"],
        chunks=all_chunks,
        verbose=True,
    )

    return {
        **state,
        "raw_results":       merged_results,
        "optimized_context": opt["context_string"],
        "all_chunks":        opt["chunks"],
    }


# ══════════════════════════════════════════════════════════════
# NODE D: GRADER (Self-Reflection)
# ══════════════════════════════════════════════════════════════
GRADER_PROMPT = """You are an AML forensics retrieval quality assessor.

Given a QUERY and the TOP RETRIEVED CHUNKS, score how well the chunks
address the query on a scale of 0.0 to 1.0:

  0.0 – 0.3 : Completely irrelevant, must retry
  0.3 – 0.6 : Partially relevant, could improve with better queries
  0.6 – 0.8 : Mostly relevant, acceptable for generation
  0.8 – 1.0 : Highly relevant, proceed to generation

Also provide brief feedback on what's missing if score < 0.7.

Respond ONLY with valid JSON:
{{"score": 0.0, "feedback": "what is missing or why score is low"}}

QUERY: {query}

TOP CHUNKS:
{chunks}"""

RELEVANCE_THRESHOLD = 0.6

def grader_node(state: AgentState) -> AgentState:
    """Score retrieved context relevance. Set should_retry flag."""
    print(f"  [GRADER] Scoring retrieval relevance...")

    # Take top 3 chunks for grading
    top_chunks = state.get("all_chunks", [])[:3]
    chunk_texts = "\n\n---\n\n".join([
        c.get("document", c.get("text", ""))[:300]
        for c in top_chunks
    ]) if top_chunks else "No chunks retrieved."

    try:
        response = llm(
            GRADER_PROMPT.format(
                query=state["query"],
                chunks=chunk_texts,
            ),
            max_tokens=200,
        )
        clean  = response.replace("```json","").replace("```","").strip()
        parsed = json.loads(clean)
        score    = float(parsed.get("score", 0.5))
        feedback = parsed.get("feedback", "")
    except Exception:
        score    = 0.7  # assume acceptable on parse error
        feedback = ""

    retry_count  = state.get("retry_count", 0)
    should_retry = (score < RELEVANCE_THRESHOLD and retry_count < MAX_RETRIES)

    print(f"  [GRADER] Score={score:.2f}, Retry={should_retry} "
          f"(attempt {retry_count+1}/{MAX_RETRIES+1})")
    if feedback:
        print(f"  [GRADER] Feedback: {feedback[:80]}")

    return {
        **state,
        "relevance_score":  score,
        "grader_feedback":  feedback,
        "should_retry":     should_retry,
        "retry_count":      retry_count + (1 if should_retry else 0),
    }


# ══════════════════════════════════════════════════════════════
# NODE E: GENERATOR
# ══════════════════════════════════════════════════════════════
def generator_node(state: AgentState) -> AgentState:
    """Generate final forensic answer using optimized context."""
    import sys
    sys.path.append(".")
    from generation.generation import ForensicsGenerator

    print(f"  [GENERATOR] Generating answer (relevance={state.get('relevance_score',0):.2f})...")

    generator = ForensicsGenerator()
    output    = generator.generate(state["query"], state["raw_results"])

    score_data = None
    if state["raw_results"].get("transactions"):
        score_data = generator.score_suspicion(
            state["raw_results"]["transactions"][0]["document"]
        )

    print(f"  [GENERATOR] Answer generated ({len(output['answer'].split())} words)")

    return {
        **state,
        "answer":          output["answer"],
        "sources":         output["sources"],
        "suspicion_score": score_data,
    }


# ══════════════════════════════════════════════════════════════
# ROUTING LOGIC
# ══════════════════════════════════════════════════════════════
def should_retry_or_generate(state: AgentState) -> Literal["query_rewriter", "generator"]:
    """Edge function: loop back to rewriter or proceed to generation."""
    if state.get("should_retry", False):
        return "query_rewriter"
    return "generator"


# ══════════════════════════════════════════════════════════════
# BUILD THE GRAPH
# ══════════════════════════════════════════════════════════════
def build_agent() -> StateGraph:
    """Construct and compile the LangGraph agent."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router",        router_node)
    graph.add_node("query_rewriter",query_rewriter_node)
    graph.add_node("retriever",     retriever_node)
    graph.add_node("grader",        grader_node)
    graph.add_node("generator",     generator_node)

    # Add edges
    graph.set_entry_point("router")
    graph.add_edge("router",         "query_rewriter")
    graph.add_edge("query_rewriter", "retriever")
    graph.add_edge("retriever",      "grader")

    # Conditional edge: retry loop or proceed
    graph.add_conditional_edges(
        "grader",
        should_retry_or_generate,
        {
            "query_rewriter": "query_rewriter",
            "generator":      "generator",
        }
    )
    graph.add_edge("generator", END)

    return graph.compile()


# ══════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════
class FFRAGAgent:
    """
    Drop-in replacement for the ForensicsRetriever + ForensicsGenerator
    combo in app.py. Uses the full agentic loop.

    Usage:
      agent = FFRAGAgent()
      result = agent.run("Find dormant accounts reactivated with large amounts")
      print(result["answer"])
    """

    def __init__(self):
        self.graph = build_agent()

    def run(self, query: str) -> dict:
        """
        Run the full agentic pipeline.

        Returns:
          {
            "answer":          str,
            "sources":         list[str],
            "suspicion_score": dict | None,
            "raw_results":     dict,
            "relevance_score": float,
            "retry_count":     int,
            "pipeline":        str,
          }
        """
        print(f"\n{'='*55}")
        print(f"  FFRAG AGENT — Processing query")
        print(f"{'='*55}")

        initial_state: AgentState = {
            "query":             query,
            "pipeline":          "both",
            "query_intent":      "conceptual",
            "rewritten_queries": [],
            "hyde_document":     "",
            "raw_results":       {},
            "optimized_context": "",
            "all_chunks":        [],
            "relevance_score":   0.0,
            "grader_feedback":   "",
            "retry_count":       0,
            "answer":            "",
            "sources":           [],
            "suspicion_score":   None,
            "should_retry":      False,
        }

        final_state = self.graph.invoke(initial_state)

        print(f"\n{'='*55}")
        print(f"  Done — retries={final_state['retry_count']}, "
              f"relevance={final_state['relevance_score']:.2f}")
        print(f"{'='*55}\n")

        return {
            "answer":          final_state["answer"],
            "sources":         final_state["sources"],
            "suspicion_score": final_state["suspicion_score"],
            "raw_results":     final_state["raw_results"],
            "relevance_score": final_state["relevance_score"],
            "retry_count":     final_state["retry_count"],
            "pipeline":        final_state["pipeline"],
        }


# ══════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    agent = FFRAGAgent()

    test_queries = [
        "What does FATF say about placement and aggregation?",
        "Find accounts that sent money to UAE",
        "Explain the SAR filing timeline for continuing structuring",
    ]

    for q in test_queries:
        result = agent.run(q)
        print(f"\nQuery:    {q}")
        print(f"Pipeline: {result['pipeline']}")
        print(f"Retries:  {result['retry_count']}")
        print(f"Relevance:{result['relevance_score']:.2f}")
        print(f"Answer:   {result['answer'][:200]}...")
        print("-" * 55)
