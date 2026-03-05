"""
setup_advanced.py
──────────────────
One-shot setup for the full Agentic GraphRAG pipeline.

Runs all 5 components in dependency order:
  1. Neo4j AuraDB ingestion     (graph_retriever.py)
  2. Semantic chunking           (semantic_chunker.py)
  3. Context optimizer test      (context_optimizer.py)
  4. LangGraph orchestrator test (langgraph_orchestrator.py)
  5. Full integration smoke test (all components together)

Run from project root:
  python setup_advanced.py

Or skip steps you've already done:
  python setup_advanced.py --skip-neo4j
  python setup_advanced.py --skip-chunking
  python setup_advanced.py --test-only
"""

import os, sys, time, argparse
sys.path.append(".")
from dotenv import load_dotenv
load_dotenv()

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):   print(f"  {GREEN}✅ {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}⚠️  {msg}{RESET}")
def err(msg):  print(f"  {RED}❌ {msg}{RESET}")
def info(msg): print(f"  {BLUE}ℹ️  {msg}{RESET}")
def header(msg):
    print(f"\n{BOLD}{'═'*60}")
    print(f"  {msg}")
    print(f"{'═'*60}{RESET}")


# ══════════════════════════════════════════════════════════════
# STEP 0 — PREFLIGHT CHECKS
# ══════════════════════════════════════════════════════════════
def preflight():
    header("STEP 0 — Preflight Checks")
    errors = []

    # Check env vars
    required = ["GROQ_API_KEY", "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"]
    for var in required:
        val = os.getenv(var, "")
        if not val or "your-" in val or "xxxxxxxx" in val:
            err(f"{var} not set in .env")
            errors.append(var)
        else:
            ok(f"{var} = {val[:20]}...")

    # Check data files
    csv_path = os.getenv("TRANSACTIONS_CSV", "data/transactions/saml_synthetic_1000.csv")
    regs_dir = os.getenv("REGULATIONS_DIR", "data/regulations")

    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        ok(f"Transactions CSV: {len(df)} rows")
    else:
        err(f"Transactions CSV not found: {csv_path}")
        errors.append("csv")

    import glob
    pdfs = glob.glob(f"{regs_dir}/*.pdf")
    if pdfs:
        ok(f"Regulations: {len(pdfs)} PDFs found")
    else:
        warn(f"No PDFs found in {regs_dir}")

    # Check ChromaDB
    chroma_dir = os.getenv("CHROMA_DIR", "chroma_db")
    if os.path.exists(chroma_dir):
        ok(f"ChromaDB found at {chroma_dir}/")
    else:
        err("ChromaDB not found — run ingestion/ingest_to_chroma.py first")
        errors.append("chroma")

    # Check pip packages
    packages = {
        "neo4j":     "pip install neo4j",
        "langgraph": "pip install langgraph",
        "networkx":  "pip install networkx",
        "pdfplumber":"pip install pdfplumber",
        "groq":      "pip install groq",
    }
    for pkg, install_cmd in packages.items():
        try:
            __import__(pkg)
            ok(f"{pkg} installed")
        except ImportError:
            err(f"{pkg} missing — run: {install_cmd}")
            errors.append(pkg)

    if errors:
        print(f"\n{RED}{BOLD}Fix the above errors before continuing.{RESET}")
        return False

    ok("All preflight checks passed")
    return True


# ══════════════════════════════════════════════════════════════
# STEP 1 — NEO4J AURADB INGESTION
# ══════════════════════════════════════════════════════════════
def setup_neo4j():
    header("STEP 1 — Neo4j AuraDB Ingestion")

    csv_path = os.getenv("TRANSACTIONS_CSV", "data/transactions/saml_synthetic_1000.csv")

    try:
        from retrieval.graph_retriever import GraphRetriever, Neo4jBackend
        info("Connecting to AuraDB...")
        t0 = time.time()

        backend = Neo4jBackend()
        backend.ingest_csv(csv_path)

        # Verify ingestion
        with backend.driver.session() as session:
            n_accounts = session.run("MATCH (a:Account) RETURN count(a) AS n").single()["n"]
            n_txns     = session.run("MATCH ()-[t:SENT]->() RETURN count(t) AS n").single()["n"]
            n_susp     = session.run("MATCH ()-[t:SENT {is_suspicious:true}]->() RETURN count(t) AS n").single()["n"]

        backend.close()
        elapsed = time.time() - t0

        ok(f"Accounts ingested:     {n_accounts}")
        ok(f"Transactions ingested: {n_txns}")
        ok(f"Suspicious flagged:    {n_susp}")
        ok(f"Ingestion time:        {elapsed:.1f}s")
        return True

    except Exception as e:
        err(f"Neo4j ingestion failed: {e}")
        warn("Falling back to NetworkX for graph queries")
        return False


# ══════════════════════════════════════════════════════════════
# STEP 2 — SEMANTIC CHUNKING
# ══════════════════════════════════════════════════════════════
def setup_semantic_chunking():
    header("STEP 2 — Semantic Chunking (Regulatory PDFs)")

    regs_dir = os.getenv("REGULATIONS_DIR", "data/regulations")
    chroma   = os.getenv("CHROMA_DIR",      "chroma_db")
    strategy = os.getenv("SEMANTIC_CHUNK_STRATEGY", "sentence_window")

    try:
        from ingestion.semantic_chunker import reingest_regulations_semantic
        t0 = time.time()

        collection = reingest_regulations_semantic(
            regulations_dir=regs_dir,
            chroma_dir=chroma,
            collection_name="regulations_v2",
            strategy=strategy,
        )

        elapsed = time.time() - t0
        ok(f"regulations_v2 collection: {collection.count()} chunks")
        ok(f"Strategy: {strategy}")
        ok(f"Time: {elapsed:.1f}s")

        # Compare with original
        try:
            import chromadb
            client = chromadb.PersistentClient(path=chroma)
            old_col = client.get_collection("regulations")
            info(f"Original regulations collection: {old_col.count()} chunks")
            info(f"New regulations_v2 collection:   {collection.count()} chunks")
        except Exception:
            pass

        return True

    except Exception as e:
        err(f"Semantic chunking failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════
# STEP 3 — CONTEXT OPTIMIZER TEST
# ══════════════════════════════════════════════════════════════
def test_context_optimizer():
    header("STEP 3 — Context Optimizer Test")

    try:
        from retrieval.context_optimizer import LostInTheMiddleReorderer

        mock_chunks = [
            {"document": "Structuring is breaking transactions below CTR thresholds.",
             "collection": "regulations", "rerank_score": 4.2},
            {"document": "Account 176667861 sent £18,346 to UAE.",
             "collection": "transactions", "rerank_score": 5.8},
            {"document": "General banking trends report Q3.",
             "collection": "regulations", "rerank_score": 0.3},
            {"document": "FATF Recommendation 19 covers high-risk jurisdictions.",
             "collection": "regulations", "rerank_score": 3.1},
            {"document": "Dormant account reactivation with £82,000 cash deposit.",
             "collection": "transactions", "rerank_score": 4.9},
        ]

        reorderer = LostInTheMiddleReorderer()
        reordered = reorderer.reorder(mock_chunks)
        context   = reorderer.format_reordered_context(reordered)

        ok(f"Reordered {len(mock_chunks)} chunks")
        ok(f"Edge[0]:  score={reordered[0]['rerank_score']} ({reordered[0]['collection']})")
        ok(f"Edge[-1]: score={reordered[-1]['rerank_score']} ({reordered[-1]['collection']})")
        ok(f"Middle:   score={reordered[len(reordered)//2]['rerank_score']} (lowest relevance)")
        ok(f"Context string: {len(context)} chars")
        return True

    except Exception as e:
        err(f"Context optimizer test failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════
# STEP 4 — LANGGRAPH AGENT TEST
# ══════════════════════════════════════════════════════════════
def test_langgraph():
    header("STEP 4 — LangGraph Orchestrator Test")

    try:
        from retrieval.langgraph_orchestrator import FFRAGAgent
        info("Building agent graph...")
        agent = FFRAGAgent()
        ok("Agent graph compiled successfully")

        # Test with one query
        info("Running test query...")
        t0     = time.time()
        result = agent.run("What is the SAR filing timeline for structuring?")
        elapsed = time.time() - t0

        ok(f"Query completed in {elapsed:.1f}s")
        ok(f"Pipeline selected:  {result['pipeline']}")
        ok(f"Retry loops:        {result['retry_count']}")
        ok(f"Relevance score:    {result['relevance_score']:.2f}")
        ok(f"Answer length:      {len(result['answer'].split())} words")
        ok(f"Sources:            {result['sources']}")

        if result["retry_count"] > 0:
            info(f"Agent self-corrected {result['retry_count']} time(s)")

        return True

    except Exception as e:
        err(f"LangGraph test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════
# STEP 5 — GRAPH RETRIEVER TEST
# ══════════════════════════════════════════════════════════════
def test_graph_retriever():
    header("STEP 5 — Graph Retriever Test")

    csv_path = os.getenv("TRANSACTIONS_CSV", "data/transactions/saml_synthetic_1000.csv")

    try:
        from retrieval.graph_retriever import GraphRetriever
        info("Initialising graph retriever...")
        gr = GraphRetriever(csv_path)

        backend_type = type(gr.backend).__name__
        ok(f"Backend: {backend_type}")

        tests = [
            ("Hub accounts",        "Find hub accounts with many incoming connections"),
            ("Round trip",          "Detect round trip transactions above £50,000"),
            ("UK→UAE corridor",     "Analyse flows from UK to UAE"),
        ]

        for label, query in tests:
            t0 = time.time()
            result = gr.query(query)
            elapsed = time.time() - t0
            ok(f"{label}: {result['n_results']} results in {elapsed:.1f}s")
            info(f"  Summary: {result['summary'][:80]}...")

        return True

    except Exception as e:
        err(f"Graph retriever test failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
def print_summary(results: dict):
    header("SETUP COMPLETE — Summary")

    all_passed = all(results.values())

    for step, passed in results.items():
        status = f"{GREEN}✅ PASS{RESET}" if passed else f"{RED}❌ FAIL{RESET}"
        print(f"  {status}  {step}")

    print()
    if all_passed:
        print(f"{GREEN}{BOLD}  All systems operational. Start the UI:{RESET}")
        print(f"  {BLUE}  streamlit run ui/app.py{RESET}\n")
        print(f"  {BLUE}  To use the agentic pipeline, update app.py:{RESET}")
        print(f"  {BLUE}  from retrieval.langgraph_orchestrator import FFRAGAgent{RESET}")
        print(f"  {BLUE}  agent = FFRAGAgent(){RESET}")
        print(f"  {BLUE}  result = agent.run(query){RESET}\n")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"{YELLOW}{BOLD}  Fix failing steps and rerun:{RESET}")
        for f in failed:
            print(f"  {RED}  → {f}{RESET}")
        print()


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="FFRAG Advanced Setup")
    parser.add_argument("--skip-neo4j",    action="store_true")
    parser.add_argument("--skip-chunking", action="store_true")
    parser.add_argument("--test-only",     action="store_true",
                        help="Skip ingestion, only run component tests")
    args = parser.parse_args()

    print(f"\n{BOLD}{'═'*60}")
    print(f"  FFRAG AGENTIC GRAPHRAG — ADVANCED SETUP")
    print(f"  Components: Neo4j · Semantic Chunks · LangGraph · GraphRAG")
    print(f"{'═'*60}{RESET}")

    if not preflight():
        sys.exit(1)

    results = {}

    if not args.test_only and not args.skip_neo4j:
        results["Neo4j AuraDB ingestion"]  = setup_neo4j()

    if not args.test_only and not args.skip_chunking:
        results["Semantic chunking"]       = setup_semantic_chunking()

    results["Context optimizer"]           = test_context_optimizer()
    results["LangGraph orchestrator"]      = test_langgraph()
    results["Graph retriever"]             = test_graph_retriever()

    print_summary(results)


if __name__ == "__main__":
    main()
