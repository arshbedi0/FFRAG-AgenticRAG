import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, json

# ── OUTPUT DIRS ──
OUT_DIR = "/data/wallet_graphs"
META_FILE = "/data/graph_metadata.json"
os.makedirs(OUT_DIR, exist_ok=True)

# ── LOAD DATA ──
df = pd.read_csv("/data/saml_synthetic_1000.csv")

# ── COUNTRY COLOR MAP ──
COUNTRY_COLORS = {
    "UK":       "#4A90D9",   # blue
    "UAE":      "#E74C3C",   # red (high risk)
    "Turkey":   "#E67E22",   # orange (high risk)
    "Morocco":  "#E74C3C",   # red (high risk)
    "Nigeria":  "#C0392B",   # dark red (high risk)
    "Mexico":   "#F39C12",   # amber
    "India":    "#27AE60",   # green
    "Pakistan": "#8E44AD",   # purple
    "Germany":  "#2ECC71",   # light green
    "USA":      "#1ABC9C",   # teal
}
DEFAULT_COLOR = "#95A5A6"

TYPOLOGY_CONFIGS = {
    "Structuring":           {"title": "Structuring Pattern",           "desc": "Repeated transfers just under £9,999 reporting threshold"},
    "Smurfing":              {"title": "Smurfing / Aggregation",        "desc": "Multiple senders funneling into one hub account"},
    "Layering":              {"title": "Layering Chain",                "desc": "Sequential large transfers through multiple hops"},
    "High_Risk_Corridor":    {"title": "High-Risk Geographic Corridor", "desc": "Cross-border flow into sanctioned/high-risk regions"},
    "Currency_Mismatch":     {"title": "Currency Mismatch Anomaly",     "desc": "Received currency inconsistent with bank location"},
    "Round_Trip":            {"title": "Round-Trip Transaction",        "desc": "Funds leaving and returning to same location"},
    "Dormant_Reactivation":  {"title": "Dormant Account Reactivation",  "desc": "Inactive account suddenly processing large transfers"},
    "Rapid_Succession":      {"title": "Rapid Succession Cluster",      "desc": "High-frequency transactions in a tight time window"},
    "Normal":                {"title": "Normal Transaction Network",    "desc": "Baseline clean transaction graph for comparison"},
}

# ─────────────────────────────────────────────
# GRAPH BUILDER
# ─────────────────────────────────────────────
def build_graph(subset_df):
    G = nx.DiGraph()
    for _, row in subset_df.iterrows():
        s = str(row["Sender_account"])
        r = str(row["Receiver_account"])
        s_loc = row["Sender_bank_location"]
        r_loc = row["Receiver_bank_location"]
        amt = row["Amount"]
        susp = row["Is_suspicious"]

        if not G.has_node(s):
            G.add_node(s, location=s_loc, total_sent=0, total_recv=0, suspicious=False)
        if not G.has_node(r):
            G.add_node(r, location=r_loc, total_sent=0, total_recv=0, suspicious=False)

        G.nodes[s]["total_sent"] += amt
        G.nodes[r]["total_recv"] += amt
        if susp:
            G.nodes[s]["suspicious"] = True
            G.nodes[r]["suspicious"] = True

        if G.has_edge(s, r):
            G[s][r]["weight"] += amt
            G[s][r]["count"] += 1
        else:
            G.add_edge(s, r, weight=amt, count=1, suspicious=bool(susp))

    return G

# ─────────────────────────────────────────────
# PLOT GRAPH
# ─────────────────────────────────────────────
def plot_graph(G, typology, filename, subtitle=""):
    cfg = TYPOLOGY_CONFIGS.get(typology, {"title": typology, "desc": ""})
    is_normal = (typology == "Normal")

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#0F1923")
    ax.set_facecolor("#0F1923")

    if len(G.nodes) == 0:
        ax.text(0.5, 0.5, "No data", color="white", ha="center", va="center")
        plt.savefig(filename, dpi=120, bbox_inches="tight")
        plt.close()
        return {}

    # Layout
    if len(G.nodes) <= 6:
        pos = nx.spring_layout(G, seed=42, k=2.5)
    elif typology == "Smurfing":
        # Star layout — find highest in-degree node as center
        hub = max(G.nodes, key=lambda n: G.in_degree(n))
        pos = nx.spring_layout(G, seed=42, k=1.8)
        pos[hub] = np.array([0.0, 0.0])
    elif typology == "Layering":
        try:
            pos = nx.kamada_kawai_layout(G)
        except:
            pos = nx.spring_layout(G, seed=42)
    else:
        pos = nx.spring_layout(G, seed=42, k=1.5, iterations=50)

    # Node attributes
    node_colors = []
    node_sizes = []
    node_borders = []

    for node in G.nodes:
        loc = G.nodes[node].get("location", "Unknown")
        susp = G.nodes[node].get("suspicious", False)
        total_flow = G.nodes[node]["total_sent"] + G.nodes[node]["total_recv"]

        color = COUNTRY_COLORS.get(loc, DEFAULT_COLOR)
        node_colors.append(color)

        size = 300 + min(total_flow / 300, 1800)
        node_sizes.append(size)
        node_borders.append("#FF4444" if susp else "#FFFFFF")

    # Edge attributes
    edges = list(G.edges(data=True))
    edge_weights = [d["weight"] for _, _, d in edges]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [0.5 + (w / max_w) * 5 for w in edge_weights]
    edge_colors = ["#FF6B6B" if d.get("suspicious") else "#4ECDC4" for _, _, d in edges]
    edge_alphas = [0.85 if d.get("suspicious") else 0.45 for _, _, d in edges]

    # Draw edges
    for i, (u, v, d) in enumerate(edges):
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            width=edge_widths[i],
            edge_color=[edge_colors[i]],
            alpha=edge_alphas[i],
            arrows=True,
            arrowsize=15,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
            ax=ax
        )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        ax=ax,
        linewidths=2,
    )

    # Node border highlight for suspicious nodes
    suspicious_nodes = [n for n in G.nodes if G.nodes[n].get("suspicious")]
    if suspicious_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=suspicious_nodes,
            node_color="none",
            node_size=[300 + min((G.nodes[n]["total_sent"]+G.nodes[n]["total_recv"])/300, 1800) + 80 for n in suspicious_nodes],
            edgecolors="#FF4444",
            linewidths=2.5,
            ax=ax
        )

    # Labels — shorten account numbers
    labels = {n: f"...{n[-4:]}" for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_color="white", ax=ax)

    # Edge amount labels on suspicious edges
    susp_edge_labels = {(u, v): f"£{d['weight']:,.0f}" for u, v, d in edges if d.get("suspicious") and d["weight"] > 5000}
    if susp_edge_labels:
        nx.draw_networkx_edge_labels(
            G, pos, susp_edge_labels,
            font_size=6.5, font_color="#FFD700",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a2e", alpha=0.7),
            ax=ax
        )

    # ── TITLE BLOCK ──
    ax.set_title(
        f"{'⚠️  SUSPICIOUS — ' if not is_normal else ''}{cfg['title']}\n{cfg['desc']}",
        color="#FF4444" if not is_normal else "#4ECDC4",
        fontsize=14, fontweight="bold", pad=16,
        fontfamily="monospace"
    )

    # ── LEGEND ──
    legend_elements = []
    seen_locs = set(nx.get_node_attributes(G, "location").values())
    for loc in seen_locs:
        color = COUNTRY_COLORS.get(loc, DEFAULT_COLOR)
        legend_elements.append(mpatches.Patch(color=color, label=loc))

    legend_elements += [
        mpatches.Patch(color="#FF6B6B", label="Suspicious flow"),
        mpatches.Patch(color="#4ECDC4", label="Normal flow"),
    ]

    legend = ax.legend(
        handles=legend_elements,
        loc="lower left", fontsize=8,
        facecolor="#1a1a2e", edgecolor="#333",
        labelcolor="white", framealpha=0.85
    )

    # ── STATS BOX ──
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    total_vol = sum(edge_weights)
    susp_vol = sum(d["weight"] for _, _, d in edges if d.get("suspicious"))
    stats_text = (
        f"Accounts: {n_nodes}   |   Transactions: {n_edges}\n"
        f"Total Volume: £{total_vol:,.0f}   |   Suspicious Vol: £{susp_vol:,.0f}"
    )
    ax.text(
        0.5, -0.04, stats_text,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=8.5, color="#AAAAAA",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", alpha=0.6)
    )

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=130, bbox_inches="tight", facecolor="#0F1923")
    plt.close()

    # Return metadata for LLaVA prompt generation
    hub_node = max(G.nodes, key=lambda n: G.degree(n))
    return {
        "typology": typology,
        "title": cfg["title"],
        "description": cfg["desc"],
        "n_accounts": n_nodes,
        "n_transactions": n_edges,
        "total_volume_gbp": round(total_vol, 2),
        "suspicious_volume_gbp": round(susp_vol, 2),
        "hub_account": f"...{hub_node[-4:]}",
        "hub_degree": G.degree(hub_node),
        "countries_involved": list(seen_locs),
        "image_path": filename,
    }

# ─────────────────────────────────────────────
# GENERATE ALL GRAPHS
# ─────────────────────────────────────────────
all_metadata = []

print("Generating wallet network graphs...\n")

# 1. One graph per suspicious typology
TYPOLOGIES = [
    "Structuring", "Smurfing", "Layering", "High_Risk_Corridor",
    "Currency_Mismatch", "Round_Trip", "Dormant_Reactivation", "Rapid_Succession"
]

for typology in TYPOLOGIES:
    subset = df[df["Type"] == typology].copy()
    # Also include some normal transactions for context
    normal_sample = df[df["Type"] == "Normal"].sample(min(10, len(df[df["Type"]=="Normal"])), random_state=42)
    combined = pd.concat([subset, normal_sample])

    G = build_graph(combined)
    fname = os.path.join(OUT_DIR, f"graph_{typology.lower()}.png")
    meta = plot_graph(G, typology, fname)
    meta["graph_id"] = f"graph_{typology.lower()}"
    all_metadata.append(meta)
    print(f"  ✓ {typology:25s} — {G.number_of_nodes()} accounts, {G.number_of_edges()} edges → {os.path.basename(fname)}")

# 2. Normal baseline graph
normal_df = df[df["Type"] == "Normal"].sample(40, random_state=99)
G_normal = build_graph(normal_df)
fname_normal = os.path.join(OUT_DIR, "graph_normal_baseline.png")
meta_normal = plot_graph(G_normal, "Normal", fname_normal)
meta_normal["graph_id"] = "graph_normal_baseline"
all_metadata.append(meta_normal)
print(f"  ✓ {'Normal Baseline':25s} — {G_normal.number_of_nodes()} accounts, {G_normal.number_of_edges()} edges → graph_normal_baseline.png")

# 3. Full network overview (all suspicious)
all_susp = df[df["Is_suspicious"] == 1]
G_all = build_graph(all_susp)
fname_all = os.path.join(OUT_DIR, "graph_full_suspicious_overview.png")
meta_all = plot_graph(G_all, "Structuring", fname_all)  # reuse color scheme
meta_all["graph_id"] = "graph_full_suspicious_overview"
meta_all["typology"] = "All_Suspicious"
meta_all["title"] = "Full Suspicious Transaction Network"
all_metadata.append(meta_all)
print(f"  ✓ {'Full Suspicious Overview':25s} — {G_all.number_of_nodes()} accounts, {G_all.number_of_edges()} edges → graph_full_suspicious_overview.png")

# ── SAVE METADATA + LLAVA PROMPTS ──
# Add LLaVA prompt template for each graph
for m in all_metadata:
    m["llava_prompt"] = (
        f"You are an AML financial forensics analyst. Analyze this transaction network graph. "
        f"The graph shows {m['n_accounts']} bank accounts connected by {m['n_transactions']} transactions "
        f"with a total volume of £{m['total_volume_gbp']:,}. "
        f"Red nodes have red borders indicating suspicious accounts. "
        f"Red/orange edges indicate suspicious transaction flows. Node size reflects transaction volume. "
        f"Describe: (1) the network topology and any hub accounts, (2) suspicious flow patterns visible, "
        f"(3) geographic corridors (node colors = countries), (4) what AML typology this resembles. "
        f"Be specific about account clustering and edge directionality."
    )

with open(META_FILE, "w") as f:
    json.dump(all_metadata, f, indent=2)

print(f"\n✅ Generated {len(all_metadata)} graphs in {OUT_DIR}")
print(f"✅ Metadata + LLaVA prompts saved to {META_FILE}")
print(f"\nFiles:")
for m in all_metadata:
    print(f"  {m['graph_id']:40s} — {m['n_accounts']} nodes, £{m['total_volume_gbp']:>12,.0f} volume")
