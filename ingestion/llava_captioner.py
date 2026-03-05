"""
llava_captioner.py
──────────────────
Reads every wallet graph PNG, sends it to LLaVA (via Ollama),
and saves the forensic captions to graph_captions.json.

REQUIREMENTS (run once on your machine):
  1. Install Ollama       → https://ollama.com/download
  2. Pull LLaVA model    → ollama pull llava
  3. pip install requests pillow

THEN RUN:
  python llava_captioner.py
"""

import os, json, base64, time
import requests
from pathlib import Path
from dotenv import load_dotenv

# ── LOAD .env from project root ──
load_dotenv()

# ── CONFIG (reads from .env, falls back to sensible defaults) ──
OLLAMA_URL     = os.getenv("OLLAMA_URL",        "http://localhost:11434/api/generate")
MODEL          = os.getenv("OLLAMA_MODEL",       "llava:7b")
GRAPHS_DIR     = os.getenv("GRAPHS_DIR",         "data/graphs")
METADATA_FILE  = os.getenv("METADATA_FILE",      "data/graphs/graph_metadata.json")
OUTPUT_FILE    = os.getenv("CAPTIONS_FILE",      "data/graph_captions.json")
TIMEOUT        = int(os.getenv("OLLAMA_TIMEOUT", "120"))

# ── LOAD METADATA ──
with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)

meta_index = {m["graph_id"]: m for m in metadata}

# ──────────────────────────────────────────────────────
# PROMPT BUILDER
# Each typology gets a slightly different forensic prompt
# so LLaVA focuses on the right visual signals
# ──────────────────────────────────────────────────────
TYPOLOGY_HINTS = {
    "Structuring": (
        "Pay close attention to: repeated edges between the same account pairs, "
        "amounts clustering just below a threshold, and whether a single account "
        "appears as both sender and receiver across many similar-sized transactions."
    ),
    "Smurfing": (
        "Pay close attention to: a central hub node with unusually high in-degree "
        "(many arrows pointing INTO it from different senders), small individual "
        "transfer amounts, and geographic diversity of sender nodes."
    ),
    "Layering": (
        "Pay close attention to: sequential chain-like paths where money hops "
        "account-to-account in a line, large individual transfer amounts, and "
        "whether the path eventually loops back or terminates."
    ),
    "High_Risk_Corridor": (
        "Pay close attention to: node colors — blue nodes are UK accounts, red/orange "
        "are high-risk countries (UAE, Turkey, Morocco, Nigeria). Identify thick edges "
        "crossing from UK (blue) to high-risk (red/orange) nodes and the £ amounts labeled."
    ),
    "Currency_Mismatch": (
        "Pay close attention to: the geographic corridor suggested by node colors, "
        "and note any edges where the payment and received currency would not match "
        "the country color of the receiving node."
    ),
    "Round_Trip": (
        "Pay close attention to: bidirectional edges or near-circular paths where "
        "money appears to leave an account cluster and return to the same cluster "
        "or same country (same node color) via a different route."
    ),
    "Dormant_Reactivation": (
        "Pay close attention to: any isolated or peripheral nodes that suddenly "
        "show a single very thick edge (large amount), suggesting an account that "
        "was dormant and just processed an unusually large transaction."
    ),
    "Rapid_Succession": (
        "Pay close attention to: tight clusters of edges between a small set of "
        "nodes, suggesting high-frequency transactions between the same accounts "
        "in a short time window."
    ),
    "Normal": (
        "This is a BASELINE normal transaction graph. Describe what a healthy, "
        "non-suspicious network looks like — sparse connections, no obvious hubs, "
        "balanced flow, no dominant corridors."
    ),
    "All_Suspicious": (
        "This is the FULL suspicious transaction network combining all typologies. "
        "Identify the most prominent clusters, the highest-degree hub accounts, "
        "dominant geographic corridors, and which areas of the graph look most alarming."
    ),
}

def build_prompt(meta: dict) -> str:
    typology  = meta.get("typology", "Unknown")
    title     = meta.get("title", typology)
    n_accs    = meta.get("n_accounts", "?")
    n_txns    = meta.get("n_transactions", "?")
    vol       = meta.get("total_volume_gbp", 0)
    susp_vol  = meta.get("suspicious_volume_gbp", 0)
    hub       = meta.get("hub_account", "unknown")
    hub_deg   = meta.get("hub_degree", "?")
    countries = ", ".join(meta.get("countries_involved", []))
    hint      = TYPOLOGY_HINTS.get(typology, "")

    return f"""You are a senior AML (Anti-Money Laundering) forensic analyst reviewing a transaction network graph.

GRAPH CONTEXT:
- Pattern type: {title}
- Accounts (nodes): {n_accs}
- Transactions (edges): {n_txns}
- Total volume: £{vol:,.0f}
- Suspicious volume: £{susp_vol:,.0f}
- Highest-degree account: {hub} (degree {hub_deg})
- Countries involved: {countries}

VISUAL ENCODING:
- Node color = bank country (Blue=UK, Red=UAE/Morocco/Nigeria, Orange=Turkey, Green=India/Germany, Purple=Pakistan)
- Node size = total transaction volume through that account
- Red border on node = flagged suspicious account
- Red/pink edges = suspicious transaction flows
- Teal/green edges = normal flows
- Edge thickness = transaction amount (thicker = larger)
- Gold labels on edges = transaction amounts in GBP

ANALYST FOCUS:
{hint}

YOUR TASK — write a structured forensic caption with these 4 sections:

1. TOPOLOGY: Describe the overall network shape (hub-spoke / chain / scattered / dense cluster).
2. SUSPICIOUS SIGNALS: List specific visual evidence of suspicious activity (hub accounts, dominant corridors, amount patterns).
3. AML TYPOLOGY MATCH: State which AML typology this most resembles and why the graph supports that conclusion.
4. RISK SUMMARY: One sentence risk verdict an analyst would write in a SAR (Suspicious Activity Report).

Be specific — reference account IDs (last 4 digits shown), amounts, and country colors you can see."""


# ──────────────────────────────────────────────────────
# IMAGE → BASE64
# ──────────────────────────────────────────────────────
def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ──────────────────────────────────────────────────────
# CALL OLLAMA / LLAVA
# ──────────────────────────────────────────────────────
def caption_image(image_path: str, prompt: str) -> str:
    img_b64 = image_to_base64(image_path)
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": [img_b64],
        "stream": True,
        "options": {
            "temperature": 0.2,
            "num_predict": 600,
        }
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT, stream=True)
        resp.raise_for_status()
        
        # Read streaming response and accumulate text
        full_response = ""
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                if "response" in data:
                    full_response += data["response"]
                if data.get("done", False):
                    break
        
        return full_response.strip()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "\n❌ Could not connect to Ollama.\n"
            "   Make sure Ollama is running:  ollama serve\n"
            "   And LLaVA is pulled:          ollama pull llava:7b"
        )
    except requests.exceptions.Timeout:
        return "[TIMEOUT — image too complex or model too slow, increase OLLAMA_TIMEOUT]"
    except Exception as e:
        return f"[ERROR: {e}]"


# ──────────────────────────────────────────────────────
# CHECK OLLAMA IS ALIVE
# ──────────────────────────────────────────────────────
def check_ollama():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        llava_available = any("llava" in m for m in models)
        print(f"✅ Ollama is running. Available models: {models}")
        if not llava_available:
            print("⚠️  LLaVA not found. Run:  ollama pull llava")
            print("   Continuing anyway — will fail on first image if not pulled.\n")
        else:
            print(f"✅ LLaVA found. Ready to caption.\n")
        return True
    except Exception:
        raise ConnectionError(
            "❌ Ollama not running. Start it with:  ollama serve\n"
            "   Then pull LLaVA:                   ollama pull llava"
        )


# ──────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────
def main():
    check_ollama()

    # Find all graph PNGs
    graph_dir = Path(GRAPHS_DIR)
    png_files  = sorted(graph_dir.glob("graph_*.png"))

    if not png_files:
        print(f"❌ No PNG files found in '{GRAPHS_DIR}/'")
        print("   Make sure you ran graph_generator.py first.")
        return

    print(f"Found {len(png_files)} graphs to caption.\n")
    print("=" * 60)

    # Load existing captions if resuming
    results = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            results = json.load(f)
        print(f"Resuming — {len(results)} captions already done.\n")

    for i, png_path in enumerate(png_files, 1):
        graph_id = png_path.stem  # e.g. "graph_smurfing"

        if graph_id in results:
            print(f"[{i}/{len(png_files)}] SKIP (already done): {graph_id}")
            continue

        # Get metadata for this graph
        meta = meta_index.get(graph_id, {"typology": "Unknown", "graph_id": graph_id})
        typology = meta.get("typology", "Unknown")

        print(f"[{i}/{len(png_files)}] Captioning: {graph_id} ({typology})")
        print(f"           Image: {png_path}")

        prompt   = build_prompt(meta)
        t_start  = time.time()
        caption  = caption_image(str(png_path), prompt)
        elapsed  = time.time() - t_start

        print(f"           ✓ Done in {elapsed:.1f}s")
        print(f"           Preview: {caption[:120]}...")
        print()

        # Store result
        results[graph_id] = {
            "graph_id":          graph_id,
            "typology":          typology,
            "title":             meta.get("title", typology),
            "image_path":        str(png_path),
            "n_accounts":        meta.get("n_accounts"),
            "n_transactions":    meta.get("n_transactions"),
            "total_volume_gbp":  meta.get("total_volume_gbp"),
            "suspicious_volume": meta.get("suspicious_volume_gbp"),
            "countries":         meta.get("countries_involved", []),
            "hub_account":       meta.get("hub_account"),
            "caption":           caption,
            "prompt_used":       prompt,
            "model":             MODEL,
            "captioned_at":      time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        # Save after every image (so crashes don't lose progress)
        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2)

    print("=" * 60)
    print(f"\n✅ All {len(results)} captions saved to {OUTPUT_FILE}")
    print(f"\nNext step: run  ingest_to_chroma.py  to embed everything into the vector DB.")

    # Print summary
    print("\nCaption length summary:")
    for gid, r in results.items():
        caption_len = len(r["caption"].split())
        status = "✓" if caption_len > 50 else "⚠️ SHORT"
        print(f"  {status} {gid:45s} — {caption_len} words")


if __name__ == "__main__":
    main()