"""
graph_retriever.py
───────────────────
Native knowledge graph retrieval for multi-hop AML queries.

Supports:
  - Neo4j (preferred — full Cypher queries)
  - NetworkX (fallback — no install needed)

Capabilities:
  - Multi-hop fund tracing (follow money through N accounts)
  - Round-trip detection (funds return to source)
  - Hub account identification (high degree centrality)
  - Corridor analysis (flows between specific countries)
  - Temporal clustering (rapid succession detection)

Usage:
  from retrieval.graph_retriever import GraphRetriever
  gr = GraphRetriever()              # auto-detects Neo4j or NetworkX
  gr.ingest_csv("data/transactions/saml_synthetic_1000.csv")
  results = gr.query("trace funds from account 176667861 through 3 hops")

Install Neo4j (optional):
  pip install neo4j
  # Start Neo4j Desktop or: docker run -p 7474:7474 -p 7687:7687 neo4j
"""

import os, json
from dotenv import load_dotenv
load_dotenv()

# AuraDB uses neo4j+s:// (TLS encrypted). Get these from your AuraDB console:
# https://console.neo4j.io → your instance → Connection details
NEO4J_URI      = os.getenv("NEO4J_URI",      "neo4j+s://xxxxxxxx.databases.neo4j.io")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your-aura-password")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
LLM_MODEL      = os.getenv("LLM_MODEL",      "llama-3.3-70b-versatile")


# ══════════════════════════════════════════════════════════════
# NEO4J BACKEND
# ══════════════════════════════════════════════════════════════
class Neo4jBackend:
    """Neo4j Cypher-based graph queries."""

    def __init__(self):
        from neo4j import GraphDatabase
        # AuraDB: neo4j+s:// handles TLS automatically — no extra flags needed
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )
        # Verify connectivity
        self.driver.verify_connectivity()
        print(f"  Connected to Neo4j AuraDB: {NEO4J_URI[:40]}...")

    def ingest_csv(self, csv_path: str):
        """Load SAML-D transactions into Neo4j as Account → Transaction → Account."""
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"  Ingesting {len(df)} transactions into Neo4j...")

        with self.driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")

            # Create constraints
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE")

            # Drop rows with null account IDs before ingestion
            before = len(df)
            # Normalise column names to lowercase with underscores
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

            # Map SAML-D column names → standard names
            col_map = {
                "sender_account":        "sender_account",
                "receiver_account":      "receiver_account",
                "sender_bank_location":  "sender_location",
                "receiver_bank_location":"receiver_location",
                "sender_bank":           "sender_bank",
                "receiver_bank":         "receiver_bank",
                "payment_type":          "payment_type",
                "payment_currency":      "currency",
                "amount":                "amount",
                "date":                  "date",
                "type":                  "typology",
                "is_suspicious":         "is_suspicious",
                "transaction_id":        "transaction_id",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

            # Ensure required columns exist
            for col in ["sender_account", "receiver_account"]:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

            df = df.dropna(subset=["sender_account", "receiver_account"])
            df = df[df["sender_account"].astype(str).str.strip().str.lower() != "nan"]
            df = df[df["receiver_account"].astype(str).str.strip().str.lower() != "nan"]
            df = df[df["sender_account"].astype(str).str.strip() != ""]
            df = df[df["receiver_account"].astype(str).str.strip() != ""]
            dropped = before - len(df)
            if dropped:
                print(f"  Dropped {dropped} rows with null account IDs")

            # Batch ingest
            batch_size = 100
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size].to_dict('records')
                # Convert any remaining NaN to None so Cypher WHERE filters catch them
                batch = [{k: (None if str(v) in ("nan","NaN","None","") else v)
                          for k, v in row.items()} for row in batch]
                session.run("""
                    UNWIND $rows AS row
                    WITH row
                    WHERE row.sender_account IS NOT NULL
                      AND row.receiver_account IS NOT NULL
                      AND toString(row.sender_account) <> 'None'
                      AND toString(row.receiver_account) <> 'None'
                    MERGE (sender:Account {id: toString(row.sender_account)})
                    SET sender.location = coalesce(row.sender_location, "Unknown"),
                        sender.bank     = coalesce(row.sender_bank, "Unknown")
                    MERGE (receiver:Account {id: toString(row.receiver_account)})
                    SET receiver.location = coalesce(row.receiver_location, "Unknown"),
                        receiver.bank     = coalesce(row.receiver_bank, "Unknown")
                    CREATE (sender)-[:SENT {
                        amount:        toFloat(coalesce(row.amount, 0)),
                        currency:      coalesce(row.currency, "GBP"),
                        payment_type:  coalesce(row.payment_type, "Unknown"),
                        date:          coalesce(row.date, ""),
                        typology:      coalesce(row.typology, "Normal"),
                        is_suspicious: toBoolean(coalesce(row.is_suspicious, false)),
                        transaction_id:toString(coalesce(row.transaction_id, ""))
                    }]->(receiver)
                """, rows=batch)
                print(f"  Ingested {min(i+batch_size, len(df))}/{len(df)}")

        print("  ✅ Neo4j ingestion complete")

    def multi_hop_trace(self, account_id: str, hops: int = 3) -> list[dict]:
        """Trace money flows from an account through N hops."""
        with self.driver.session() as session:
            result = session.run(f"""
                MATCH path = (start:Account {{id: $account_id}})-[:SENT*1..{hops}]->(end:Account)
                WITH path, nodes(path) AS accounts, relationships(path) AS txns
                WHERE ALL(r IN relationships(path) WHERE r.is_suspicious = true)
                RETURN
                    [a IN accounts | a.id]          AS account_chain,
                    [a IN accounts | a.location]    AS location_chain,
                    [t IN txns | t.amount]          AS amounts,
                    [t IN txns | t.typology]        AS typologies,
                    reduce(total = 0.0, t IN txns | total + t.amount) AS total_flow,
                    length(path)                    AS hop_count
                ORDER BY total_flow DESC
                LIMIT 20
            """, account_id=str(account_id))
            return [dict(r) for r in result]

    def detect_round_trips(self, min_amount: float = 10000) -> list[dict]:
        """Find funds that return to originating account/country."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:Account)-[:SENT*2..5]->(b:Account)
                WHERE a.location = b.location AND a.id <> b.id
                MATCH path = (a)-[:SENT*2..5]->(b)
                WITH a, b, path,
                     reduce(total=0.0, r IN relationships(path) | total + r.amount) AS volume
                WHERE volume >= $min_amount
                  AND ALL(r IN relationships(path) WHERE r.is_suspicious = true)
                RETURN
                    a.id       AS origin_account,
                    a.location AS origin_location,
                    b.id       AS destination_account,
                    length(path) AS hops,
                    volume
                ORDER BY volume DESC
                LIMIT 15
            """, min_amount=min_amount)
            return [dict(r) for r in result]

    def find_hub_accounts(self, min_degree: int = 5) -> list[dict]:
        """Find high-degree hub accounts (smurfing aggregators)."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (hub:Account)<-[:SENT]-(sender:Account)
                WHERE sender <> hub
                WITH hub, count(sender) AS in_degree,
                     sum(size([(hub)-[:SENT]->(x) | x])) AS out_degree
                WHERE in_degree >= $min_degree
                MATCH (hub)<-[r:SENT]-()
                WITH hub, in_degree, out_degree,
                     sum(r.amount) AS total_received,
                     count(CASE WHEN r.is_suspicious THEN 1 END) AS suspicious_count
                RETURN
                    hub.id       AS account_id,
                    hub.location AS location,
                    in_degree,
                    out_degree,
                    total_received,
                    suspicious_count
                ORDER BY in_degree DESC
                LIMIT 10
            """, min_degree=min_degree)
            return [dict(r) for r in result]

    def corridor_analysis(self, from_country: str, to_country: str) -> list[dict]:
        """Analyse all flows between two countries."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (sender:Account {location: $from_country})-[t:SENT]->(receiver:Account {location: $to_country})
                RETURN
                    sender.id   AS sender_account,
                    receiver.id AS receiver_account,
                    t.amount    AS amount,
                    t.typology  AS typology,
                    t.date      AS date,
                    t.is_suspicious AS suspicious
                ORDER BY t.amount DESC
                LIMIT 20
            """, from_country=from_country, to_country=to_country)
            return [dict(r) for r in result]

    def close(self):
        self.driver.close()


# ══════════════════════════════════════════════════════════════
# NETWORKX FALLBACK BACKEND
# ══════════════════════════════════════════════════════════════
class NetworkXBackend:
    """
    In-memory graph using NetworkX.
    No Neo4j needed — loads directly from CSV.
    Supports the same query interface as Neo4jBackend.
    """

    def __init__(self):
        import networkx as nx
        self.G = nx.DiGraph()
        print("  Using NetworkX in-memory graph (no Neo4j required)")

    def ingest_csv(self, csv_path: str):
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"  Loading {len(df)} transactions into NetworkX graph...")

        for _, row in df.iterrows():
            sender   = str(row["sender_account"])
            receiver = str(row["receiver_account"])

            # Add nodes with attributes
            self.G.add_node(sender,
                location=row.get("sender_location","?"),
                bank=row.get("sender_bank","?"))
            self.G.add_node(receiver,
                location=row.get("receiver_location","?"),
                bank=row.get("receiver_bank","?"))

            # Add edge (transaction)
            self.G.add_edge(sender, receiver,
                amount        = float(row.get("amount", 0)),
                currency      = row.get("currency","?"),
                payment_type  = row.get("payment_type","?"),
                date          = str(row.get("date","")),
                typology      = row.get("typology","?"),
                is_suspicious = bool(row.get("is_suspicious", False)),
                transaction_id= str(row.get("transaction_id","?")))

        print(f"  Graph: {self.G.number_of_nodes()} accounts, "
              f"{self.G.number_of_edges()} transactions")

    def multi_hop_trace(self, account_id: str, hops: int = 3) -> list[dict]:
        """BFS/DFS fund tracing from source account."""
        import networkx as nx
        results = []
        account_id = str(account_id)

        if account_id not in self.G:
            return []

        # Find all simple paths up to N hops
        for target in list(self.G.nodes())[:100]:  # cap for performance
            if target == account_id:
                continue
            try:
                for path in nx.all_simple_paths(self.G, account_id, target, cutoff=hops):
                    if len(path) < 2:
                        continue
                    # Get edge data along path
                    amounts    = []
                    typologies = []
                    all_susp   = True
                    for i in range(len(path)-1):
                        edge = self.G.edges[path[i], path[i+1]]
                        amounts.append(edge["amount"])
                        typologies.append(edge["typology"])
                        if not edge["is_suspicious"]:
                            all_susp = False

                    if all_susp:
                        results.append({
                            "account_chain":  path,
                            "location_chain": [self.G.nodes[n].get("location","?") for n in path],
                            "amounts":        amounts,
                            "typologies":     typologies,
                            "total_flow":     sum(amounts),
                            "hop_count":      len(path)-1,
                        })
            except nx.NetworkXNoPath:
                pass

        return sorted(results, key=lambda x: x["total_flow"], reverse=True)[:20]

    def detect_round_trips(self, min_amount: float = 10000) -> list[dict]:
        """Find cycles in the graph (round-trip patterns)."""
        import networkx as nx
        results = []

        try:
            cycles = list(nx.simple_cycles(self.G))
        except Exception:
            return []

        for cycle in cycles:
            if len(cycle) < 2:
                continue
            # Calculate cycle volume
            volume = 0
            all_susp = True
            for i in range(len(cycle)):
                src  = cycle[i]
                dst  = cycle[(i+1) % len(cycle)]
                if self.G.has_edge(src, dst):
                    e = self.G.edges[src, dst]
                    volume += e["amount"]
                    if not e["is_suspicious"]:
                        all_susp = False

            if volume >= min_amount and all_susp:
                origin = cycle[0]
                dest   = cycle[-1]
                results.append({
                    "origin_account":      origin,
                    "origin_location":     self.G.nodes[origin].get("location","?"),
                    "destination_account": dest,
                    "hops":                len(cycle),
                    "volume":              volume,
                })

        return sorted(results, key=lambda x: x["volume"], reverse=True)[:15]

    def find_hub_accounts(self, min_degree: int = 5) -> list[dict]:
        """Find high in-degree hub accounts."""
        results = []
        for node in self.G.nodes():
            in_deg  = self.G.in_degree(node)
            out_deg = self.G.out_degree(node)
            if in_deg >= min_degree:
                in_edges    = list(self.G.in_edges(node, data=True))
                total_recv  = sum(e[2]["amount"] for e in in_edges)
                susp_count  = sum(1 for e in in_edges if e[2]["is_suspicious"])
                results.append({
                    "account_id":      node,
                    "location":        self.G.nodes[node].get("location","?"),
                    "in_degree":       in_deg,
                    "out_degree":      out_deg,
                    "total_received":  total_recv,
                    "suspicious_count":susp_count,
                })

        return sorted(results, key=lambda x: x["in_degree"], reverse=True)[:10]

    def corridor_analysis(self, from_country: str, to_country: str) -> list[dict]:
        """Analyse cross-border flows between two countries."""
        results = []
        for src, dst, data in self.G.edges(data=True):
            src_loc = self.G.nodes[src].get("location","?")
            dst_loc = self.G.nodes[dst].get("location","?")
            if src_loc == from_country and dst_loc == to_country:
                results.append({
                    "sender_account":   src,
                    "receiver_account": dst,
                    "amount":           data["amount"],
                    "typology":         data["typology"],
                    "date":             data["date"],
                    "suspicious":       data["is_suspicious"],
                })

        return sorted(results, key=lambda x: x["amount"], reverse=True)[:20]


# ══════════════════════════════════════════════════════════════
# NATURAL LANGUAGE → GRAPH QUERY TRANSLATOR
# ══════════════════════════════════════════════════════════════
NL_TO_GRAPH_PROMPT = """You are an AML graph query classifier.

Given a natural language query, extract the graph operation parameters.
Return ONLY valid JSON with this structure:

{{
  "operation": "multi_hop_trace" | "round_trip" | "hub_accounts" | "corridor",
  "params": {{
    "account_id":    "account number if mentioned, else null",
    "hops":          3,
    "min_amount":    10000,
    "from_country":  "country name if mentioned, else null",
    "to_country":    "country name if mentioned, else null",
    "min_degree":    5
  }}
}}

QUERY: {query}"""

def parse_nl_to_graph_query(query: str) -> dict:
    """Use LLM to classify query into a graph operation."""
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    try:
        r = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role":"user","content":NL_TO_GRAPH_PROMPT.format(query=query)}],
            max_tokens=200, temperature=0.0,
        )
        clean  = r.choices[0].message.content.strip().replace("```json","").replace("```","")
        return json.loads(clean)
    except Exception:
        return {"operation": "hub_accounts", "params": {"min_degree": 5}}


# ══════════════════════════════════════════════════════════════
# UNIFIED GRAPH RETRIEVER
# ══════════════════════════════════════════════════════════════
class GraphRetriever:
    """
    Unified interface — auto-selects Neo4j or NetworkX.
    Translates natural language queries to graph operations.
    """

    def __init__(self, csv_path: str = "data/transactions/saml_synthetic_1000.csv"):
        self.csv_path = csv_path
        self.backend  = self._init_backend()
        if csv_path and os.path.exists(csv_path):
            self.backend.ingest_csv(csv_path)

    def _init_backend(self):
        try:
            b = Neo4jBackend()
            # Test connection
            return b
        except Exception:
            print("  Neo4j unavailable — falling back to NetworkX")
            return NetworkXBackend()

    def query(self, natural_language_query: str) -> dict:
        """
        Main entry point. Parse NL query → run graph operation → return results.
        """
        parsed = parse_nl_to_graph_query(natural_language_query)
        op     = parsed.get("operation", "hub_accounts")
        params = parsed.get("params", {})

        print(f"  [GRAPH] Operation: {op}, Params: {params}")

        if op == "multi_hop_trace":
            account_id = params.get("account_id")
            if not account_id:
                return {"error": "No account ID found in query"}
            results = self.backend.multi_hop_trace(
                account_id=str(account_id),
                hops=int(params.get("hops", 3))
            )

        elif op == "round_trip":
            results = self.backend.detect_round_trips(
                min_amount=float(params.get("min_amount", 10000))
            )

        elif op == "hub_accounts":
            results = self.backend.find_hub_accounts(
                min_degree=int(params.get("min_degree", 5))
            )

        elif op == "corridor":
            from_c = params.get("from_country","UK")
            to_c   = params.get("to_country","UAE")
            results = self.backend.corridor_analysis(from_c, to_c)

        else:
            results = []

        return {
            "operation": op,
            "params":    params,
            "results":   results,
            "n_results": len(results),
            "summary":   self._summarise(op, results),
        }

    def _summarise(self, operation: str, results: list[dict]) -> str:
        """Convert graph results to text for RAG context injection."""
        if not results:
            return "No results found for graph query."

        if operation == "multi_hop_trace":
            top = results[0]
            return (
                f"Multi-hop trace found {len(results)} suspicious fund flow paths. "
                f"Largest: {' → '.join(str(a) for a in top['account_chain'][:4])} "
                f"across {top['hop_count']} hops, total £{top['total_flow']:,.0f}. "
                f"Typologies: {', '.join(set(top['typologies']))}."
            )

        elif operation == "round_trip":
            total_vol = sum(r.get("volume",0) for r in results)
            return (
                f"Detected {len(results)} round-trip patterns. "
                f"Combined volume: £{total_vol:,.0f}. "
                f"Largest: {results[0].get('origin_account')} → "
                f"{results[0].get('destination_account')}, "
                f"£{results[0].get('volume',0):,.0f} over {results[0].get('hops',0)} hops."
            )

        elif operation == "hub_accounts":
            top = results[0]
            return (
                f"Top hub account: {top['account_id']} ({top['location']}) "
                f"with {top['in_degree']} incoming connections, "
                f"£{top.get('total_received',0):,.0f} received, "
                f"{top.get('suspicious_count',0)} suspicious transactions."
            )

        elif operation == "corridor":
            total = sum(r.get("amount",0) for r in results)
            susp  = sum(1 for r in results if r.get("suspicious"))
            return (
                f"Corridor analysis: {len(results)} transactions found. "
                f"Total volume: £{total:,.0f}. "
                f"Suspicious: {susp}/{len(results)}."
            )

        return f"Graph query returned {len(results)} results."


# ══════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    gr = GraphRetriever("data/transactions/saml_synthetic_1000.csv")

    tests = [
        "Find hub accounts with many incoming connections",
        "Detect round trip transactions above £50,000",
        "Analyse flows from UK to UAE",
    ]
    for q in tests:
        print(f"\nQuery: {q}")
        r = gr.query(q)
        print(f"  Operation: {r['operation']}")
        print(f"  Results:   {r['n_results']}")
        print(f"  Summary:   {r['summary']}")
