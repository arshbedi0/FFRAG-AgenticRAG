"""
mock_captions.py
────────────────
Generates realistic mock LLaVA-style captions for all graphs.
Use this to test your full RAG pipeline BEFORE Ollama is set up.
When you run the real llava_captioner.py, it will OVERWRITE these.

RUN:  python mock_captions.py
"""

import json, time
from pathlib import Path

METADATA_FILE = "graph_metadata.json"
OUTPUT_FILE   = "graph_captions.json"

with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)

MOCK_CAPTIONS = {
    "graph_structuring": """
1. TOPOLOGY: The network shows a moderately dense directed graph with 61 accounts. Several account pairs display multiple parallel edges, indicating repeated transactions between the same sender-receiver combinations. No single dominant hub, but clusters of 3-4 accounts appear tightly interconnected.

2. SUSPICIOUS SIGNALS: Account ...8745 appears as a sender in 9 separate transactions, all with amounts between £8,612 and £9,987 — consistently below the £10,000 reporting threshold. Edge thicknesses are deliberately uniform, suggesting intentional amount calibration. Cash and cheque payment types dominate the suspicious edges (red).

3. AML TYPOLOGY MATCH: Classic Structuring (also known as smurfing in some jurisdictions). The deliberate fragmentation of what could be single large transfers into sub-threshold amounts is the defining visual signature. The consistency of amounts just under £9,999 is statistically improbable in legitimate commerce.

4. RISK SUMMARY: Account ...8745 is assessed HIGH RISK for structuring — 9 transactions totaling £86,340 were deliberately kept below the £10,000 SAR reporting threshold across a 48-hour window, consistent with FATF Recommendation 20 violation indicators.
""",
    "graph_smurfing": """
1. TOPOLOGY: Clear hub-and-spoke architecture. Account ...8098 sits at the center of a star formation with 11 inbound edges from geographically diverse sender accounts (UAE, Morocco, Nigeria, UK, Turkey). Outbound edges from the hub are fewer and larger, indicating aggregation before onward transfer.

2. SUSPICIOUS SIGNALS: The hub account ...8098 has an in-degree of 11 vs out-degree of 3 — a 3.7:1 ratio strongly indicative of aggregation. Individual inbound amounts range £500–£2,400 (small, below radar), while outbound transfers from the hub reach £18,000–£31,000. Red-bordered nodes on all sender accounts confirm they are flagged as suspicious participants.

3. AML TYPOLOGY MATCH: Textbook Smurfing / Aggregation. Multiple low-value transfers from distributed sources funnel into one account before a consolidated exit transfer — the classic placement-to-layering transition in the money laundering cycle.

4. RISK SUMMARY: Account ...8098 is assessed CRITICAL RISK as a smurfing aggregator receiving £50,924 from 11 separate accounts across 5 jurisdictions before executing consolidated outbound transfers, consistent with FATF Recommendation 16 wire transfer indicators.
""",
    "graph_layering": """
1. TOPOLOGY: The graph displays a sequential chain structure — several linear paths of 4-6 accounts where funds pass account-to-account in a directed sequence before dispersing. Unlike hub-spoke patterns, no single node dominates; instead, the graph resembles multiple parallel pipelines.

2. SUSPICIOUS SIGNALS: The dominant red edges trace £20,000–£75,000 ACH transfers moving through chains: ...3730 → ...9965 → ...4069 → ...6312. Each hop reduces traceability. Total suspicious volume is £1,728,144 — the highest of any typology graph — concentrated in just 35 transactions. Edge labels confirm individual transfers of £35,000–£67,000.

3. AML TYPOLOGY MATCH: Layering — the second stage of classic money laundering. Large amounts are moved rapidly through sequential accounts specifically to obscure the audit trail from the original source. The use of ACH transfers (fast, low-fee) accelerates the layering velocity.

4. RISK SUMMARY: A layering chain involving accounts ...3730 through ...6312 moved £1.2M across 6 hops in under 4 hours, consistent with professional money laundering operations and triggering FATF Recommendation 10 CDD obligations for all intermediate institutions.
""",
    "graph_high_risk_corridor": """
1. TOPOLOGY: Bipartite-style structure with two clearly visible clusters separated by a geographic boundary. Left cluster: UK accounts (blue nodes). Right cluster: UAE, Morocco, and Turkey accounts (red/orange nodes). Heavy red edges cross the center, forming a dense UK-to-high-risk corridor.

2. SUSPICIOUS SIGNALS: Accounts ...2384 and ...8745 (UK, blue) each send to 8+ receivers in UAE and Morocco. Edge labels show individual transfers of £17,000–£46,000. The directionality is almost entirely one-way — UK outbound, high-risk inbound. Suspicious volume is £927,873 out of £955,673 total — 97% of all volume in this graph is suspicious.

3. AML TYPOLOGY MATCH: High-Risk Geographic Corridor. The visual concentration of thick red edges from UK (FATF member, strong AML regime) to UAE/Morocco/Turkey (historically higher-risk jurisdictions for trade-based money laundering) is a primary red flag for cross-border layering and potential sanctions evasion.

4. RISK SUMMARY: Two UK-based hub accounts transferred £927,873 to receivers across UAE, Morocco, and Turkey via 35 cross-border transactions, representing a HIGH-RISK corridor consistent with trade-based money laundering and warranting enhanced due diligence under FATF Recommendation 19.
""",
    "graph_currency_mismatch": """
1. TOPOLOGY: Moderately scattered graph with 60 accounts. No dominant hub but several mid-degree nodes acting as pass-through points. Geographic diversity is high — 8 countries represented by node colors.

2. SUSPICIOUS SIGNALS: Multiple edges connect nodes where the sending country color (e.g. UK blue) routes to receiving nodes (UAE red) but the received currency metadata indicates UK pounds rather than Dirhams. This currency-location mismatch appears in approximately 60% of the red edges, suggesting deliberate currency obfuscation. Account ...4579 receives from 6 different countries while its declared received currency is inconsistent with its bank location.

3. AML TYPOLOGY MATCH: Currency Mismatch Anomaly — a sub-pattern of trade-based money laundering where mismatched currencies between sender/receiver countries indicate either fictitious invoicing or deliberate value transfer obfuscation. Often used to exploit gaps between banking system records and actual currency flows.

4. RISK SUMMARY: Account ...4579 and 12 associated accounts show systematic currency-location mismatches across £155,976 in transactions, consistent with trade-based money laundering indicators under FATF Guidance on Trade-Based Money Laundering.
""",
    "graph_round_trip": """
1. TOPOLOGY: The graph contains notable circular subgraphs — paths where funds leave a node cluster and return to the same geographic color zone (same country) via 2-3 intermediate accounts. Several near-loops are visible where edges form partial circuits.

2. SUSPICIOUS SIGNALS: Account ...7492 sends £28,000 to account ...3839 (different country), which forwards to ...9965, which sends back to a UK-blue account in the same cluster as the originator. The round-trip path is completed in 3 hops. Multiple similar circular patterns exist in the graph. Suspicious volume of £1,357,924 suggests this is a high-value operation.

3. AML TYPOLOGY MATCH: Round-Trip Transaction. Money exits the originating account, passes through intermediaries (often in different jurisdictions to create paper complexity), and returns to a beneficial owner-connected account. Classic integration-stage laundering to make criminal funds appear as legitimate business returns or investments.

4. RISK SUMMARY: At least 3 distinct round-trip circuits were identified totaling £1.35M, where funds returned to originating-country accounts after transiting through UAE and Turkish intermediaries, consistent with integration-stage laundering under FATF Typologies Report on Round-Tripping.
""",
    "graph_dormant_reactivation": """
1. TOPOLOGY: The graph shows 81 accounts with a distinctive pattern: several peripheral, low-degree nodes (dormant accounts) each connected by a single very thick edge to the main network. These dormant nodes had no prior activity and appear as isolated satellites suddenly firing large transfers.

2. SUSPICIOUS SIGNALS: Account ...5113 shows a single inbound edge of £87,000 from a previously isolated node — the edge thickness is the largest in the graph by a significant margin. Three similar dormant-activation events are visible: accounts with degree-1 connections all carrying amounts £30,000–£97,000. The sudden activation pattern with no preceding small test transactions is anomalous.

3. AML TYPOLOGY MATCH: Dormant Account Reactivation — a common placement technique where accounts are kept inactive to avoid transaction monitoring thresholds, then suddenly activated for large single transfers. Often associated with the initial placement stage of laundering cash proceeds.

4. RISK SUMMARY: Four previously dormant accounts were reactivated with single transactions ranging £30,000–£97,000 totaling £2.2M, representing a HIGH-RISK placement indicator requiring immediate SAR filing under POCA 2002 Section 330 obligations.
""",
    "graph_rapid_succession": """
1. TOPOLOGY: Dense micro-clusters visible throughout the graph — tight groups of 4-6 accounts with numerous edges between them. Unlike other patterns, edge thicknesses are moderate and relatively uniform, but the sheer edge count between small account clusters stands out.

2. SUSPICIOUS SIGNALS: Accounts ...3839, ...8098, and ...7629 form a tight triangle with 8 edges between them in both directions — suggesting rapid back-and-forth transfers. The time-compressed nature (all within a short window) is implied by the cluster density. Red edges dominate the central cluster. Individual amounts are modest (£1,000–£8,000) but frequency is extreme.

3. AML TYPOLOGY MATCH: Rapid Succession — high-frequency transaction cycling between a small account cluster. Often used to generate artificial transaction volume to justify the movement of larger consolidated sums, or to overwhelm transaction monitoring systems with noise.

4. RISK SUMMARY: A cluster of 6 accounts executed £193,731 across 35 transactions in a tight timeframe with amounts deliberately varied to avoid pattern detection, consistent with velocity-abuse typologies described in the FCA's Financial Crime Guide Section 3.2.
""",
    "graph_normal_baseline": """
1. TOPOLOGY: Sparse, decentralised graph with 74 accounts forming small disconnected subgraphs of 2-4 nodes each. No dominant hub. No geographic corridor dominance. The overall structure resembles what you would expect from legitimate retail banking — individual bilateral transactions with no clustering.

2. SUSPICIOUS SIGNALS: None identified. All edges are teal (normal flow). No red borders on any nodes. No red edges. Amount labels are absent (no single transaction is large enough to warrant labeling). Node sizes are small and relatively uniform, indicating balanced, low-volume individual transactions.

3. AML TYPOLOGY MATCH: None. This graph represents the expected baseline appearance of a clean transaction network. The absence of hub formation, geographic concentration, or unusual edge patterns is precisely what distinguishes it from all suspicious typology graphs.

4. RISK SUMMARY: Network assessed LOW RISK — transaction patterns show no AML red flags, with decentralised flow, balanced amounts, and no suspicious geographic corridors, consistent with normal retail customer behaviour.
""",
    "graph_full_suspicious_overview": """
1. TOPOLOGY: Large complex network of 198 accounts with £6.8M total volume. Multiple distinct cluster types are visible simultaneously — hub-spoke formations in the upper-left (smurfing), chain structures in the centre (layering), dense bilateral clusters (rapid succession), and clear geographic corridors flowing rightward toward high-risk country nodes.

2. SUSPICIOUS SIGNALS: The graph is overwhelmingly red — approximately 85% of all edges are suspicious flows. At least 4 major hub accounts are identifiable by node size and in-degree. The UK-to-UAE/Turkey corridor is the dominant geographic feature, with thick red edge bundles crossing the centre. Several dormant-node activations are visible as peripheral satellites with single thick edges. Total suspicious volume across all typologies: £6.8M.

3. AML TYPOLOGY MATCH: Multi-typology network — this overview graph captures all 8 AML typologies simultaneously, representing a sophisticated criminal network that employs multiple techniques in combination to maximise laundering efficiency and evade any single detection mechanism.

4. RISK SUMMARY: The full suspicious network represents a CRITICAL RISK enterprise-level laundering operation spanning 8 typologies, 10 jurisdictions, and £6.8M in suspicious flows — requiring coordinated SAR filing, network-wide account freezing, and referral to the National Crime Agency under POCA 2002 Section 337.
"""
}

def main():
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)

    meta_index = {m["graph_id"]: m for m in metadata}
    results = {}

    print("Generating mock LLaVA captions...\n")

    for graph_id, caption in MOCK_CAPTIONS.items():
        meta = meta_index.get(graph_id, {})
        results[graph_id] = {
            "graph_id":          graph_id,
            "typology":          meta.get("typology", "Unknown"),
            "title":             meta.get("title", graph_id),
            "image_path":        f"wallet_graphs/{graph_id}.png",
            "n_accounts":        meta.get("n_accounts"),
            "n_transactions":    meta.get("n_transactions"),
            "total_volume_gbp":  meta.get("total_volume_gbp"),
            "suspicious_volume": meta.get("suspicious_volume_gbp"),
            "countries":         meta.get("countries_involved", []),
            "hub_account":       meta.get("hub_account"),
            "caption":           caption.strip(),
            "model":             "mock-for-testing",
            "captioned_at":      time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        print(f"  ✓ {graph_id}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ {len(results)} mock captions saved to {OUTPUT_FILE}")
    print("   Replace with real captions by running llava_captioner.py once Ollama is set up.")

if __name__ == "__main__":
    main()
