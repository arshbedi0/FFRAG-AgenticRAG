"""
app.py — Financial Forensics RAG Chatbot UI (Streamlit Cloud Ready)
Run: streamlit run ui/app.py
"""

import os
import sys
import subprocess
import streamlit as st
from pathlib import Path
import sys  # Add this at the top of app.py
# ... inside your auto-build logic ...
if not os.path.exists("chroma_db"):
    with st.spinner("🏗️ Building Forensic Database..."):
        # sys.executable is the magic path to the correct Python environment
        result = subprocess.run(
            [sys.executable, "ingestion/ingest_to_chroma.py"], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode != 0:
            st.error(f"❌ Ingestion Failed: {result.stderr}")
            st.stop()
        else:
            st.success("✅ Database Built!")
# ─── INITIALIZE CHROMADB ON STREAMLIT CLOUD ──────────────────────────────────
# This runs ONCE on first load, then uses cached data on subsequent visits

@st.cache_resource(show_spinner=False)
def initialize_chroma_db():
    """Initialize ChromaDB: download from GitHub or build locally."""
    chroma_dir = Path("chroma_db")
    chroma_sqlite = chroma_dir / "chroma.sqlite3"
    
    # Already initialized
    if chroma_sqlite.exists():
        print(f"✅ ChromaDB already initialized at {chroma_dir}")
        return True
    
    # Try to download from GitHub Release
    print("📥 ChromaDB not found. Attempting to download from GitHub...")
    try:
        import urllib.request
        import zipfile
        
        download_url = "https://github.com/arshbedi0/FFRAG-AgenticRAG/releases/download/v1.0/chroma_db.zip"
        zip_file = "chroma_db_temp.zip"
        
        with st.spinner("📥 Downloading ChromaDB (~2 min, ~150 MB)..."):
            urllib.request.urlretrieve(download_url, zip_file)
            print(f"   ✅ Downloaded {zip_file}")
        
        with st.spinner("📦 Extracting ChromaDB..."):
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(".")
            print(f"   ✅ Extracted successfully")
        
        # Cleanup
        if os.path.exists(zip_file):
            os.remove(zip_file)
        
        if chroma_sqlite.exists():
            st.success("✅ ChromaDB ready! (Downloaded from GitHub)")
            return True
    
    except Exception as e:
        print(f"⚠️  GitHub download failed: {e}")
    
    # Fallback: Build locally
    print("🔨 Building ChromaDB locally from source data...")
    try:
        with st.spinner("🔨 Building ChromaDB from source (~3 min, requires data files)..."):
            result = subprocess.run(
                ["python", "ingestion/ingest_to_chroma.py"],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                print(f"Build error: {result.stderr}")
                st.error(f"❌ Failed to build database: {result.stderr}")
                return False
        
        if chroma_sqlite.exists():
            st.success("✅ ChromaDB built successfully from source!")
            return True
    
    except subprocess.TimeoutExpired:
        st.error("❌ ChromaDB build timed out (took too long)")
        return False
    except Exception as e:
        st.error(f"❌ Error building database: {str(e)}")
        return False
    
    return False


# Initialize on app startup
if not Path("chroma_db/chroma.sqlite3").exists():
    chroma_ready = initialize_chroma_db()
    if not chroma_ready:
        st.error("🚨 ChromaDB failed to initialize. App cannot run.")
        st.stop()

# ... rest of your app.py code ...
# ... rest of your app.py code ...
import sys, os, random, re
sys.path.append(".")

import streamlit as st
import json, uuid, datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

SESSIONS_DIR = Path("chat_sessions")
SESSIONS_DIR.mkdir(exist_ok=True)

def save_session(session_id, name, messages):
    """Persist a session to disk as JSON."""
    path = SESSIONS_DIR / f"{session_id}.json"
    data = {
        "id":       session_id,
        "name":     name,
        "created":  str(datetime.datetime.now()),
        "messages": [
            {k: v for k, v in m.items() if k != "retrieval"}
            for m in messages
        ]
    }
    path.write_text(json.dumps(data, indent=2, default=str))

def load_all_sessions():
    """Return list of {id, name, created, preview} sorted newest first."""
    sessions = []
    for p in sorted(SESSIONS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            d = json.loads(p.read_text())
            # Preview = first user message
            preview = next((m["content"][:50] for m in d.get("messages",[]) if m["role"]=="user"), "Empty session")
            sessions.append({"id": d["id"], "name": d["name"], "created": d["created"], "preview": preview})
        except Exception:
            pass
    return sessions

def load_session(session_id):
    """Load messages from a saved session."""
    path = SESSIONS_DIR / f"{session_id}.json"
    if path.exists():
        return json.loads(path.read_text()).get("messages", [])
    return []

def delete_session(session_id):
    path = SESSIONS_DIR / f"{session_id}.json"
    if path.exists():
        path.unlink()

def auto_name(messages):
    """Generate session name from first user query."""
    for m in messages:
        if m["role"] == "user":
            q = m["content"]
            return q[:40] + ("..." if len(q) > 40 else "")
    return "New Session"

st.set_page_config(
    page_title="FFRAG — Financial Forensics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; background-color: #0a0e1a; color: #c8d0e0; }
.main { background-color: #0a0e1a; }
.block-container { padding: 2rem 2rem 4rem; max-width: 1200px; }
.ffrag-logo { font-family: 'IBM Plex Mono', monospace; font-size: 28px; font-weight: 600; color: #4a9eff; letter-spacing: -1px; }
.ffrag-subtitle { font-size: 13px; color: #556b8a; letter-spacing: 2px; text-transform: uppercase; }
.user-bubble { background: #0f1e38; border: 1px solid #1e3a5f; border-radius: 12px 12px 2px 12px; padding: 14px 18px; margin: 8px 0; font-size: 14px; color: #a8c0e0; }
.assistant-bubble { background: #080d1a; border: 1px solid #152035; border-radius: 2px 12px 12px 12px; padding: 18px 20px; margin: 8px 0; font-size: 14px; line-height: 1.7; }
.source-badges { display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0 0; }
.badge { font-family: 'IBM Plex Mono', monospace; font-size: 10px; padding: 3px 10px; border-radius: 20px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; }
.badge-txn { background: #0f2a1e; color: #3ddc84; border: 1px solid #1f5c3a; }
.badge-graph { background: #1a1a0f; color: #f0c040; border: 1px solid #4a3c00; }
.badge-reg { background: #1a0f1a; color: #c084f0; border: 1px solid #4a1a6a; }
.score-block { display: inline-flex; align-items: center; gap: 10px; background: #0c1525; border: 1px solid #1e3050; border-radius: 8px; padding: 8px 16px; margin-top: 12px; font-family: 'IBM Plex Mono', monospace; font-size: 12px; }
.score-critical { color: #ff4444; border-color: #4a0000; }
.score-high { color: #ff8c42; border-color: #3a1a00; }
.score-medium { color: #f0c040; border-color: #3a2a00; }
.score-low { color: #3ddc84; border-color: #003a20; }
.guardrail-block { background: #1a0f0f; border: 1px solid #4a1a1a; border-left: 3px solid #ff4444; border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 8px 0; font-size: 13px; color: #cc8888; }
.warning-block { background: #0c1a10; border: 1px solid #1a4020; border-left: 3px solid #3ddc84; border-radius: 0 8px 8px 0; padding: 10px 14px; font-size: 12px; color: #5aaa70; }
.tip-box { background: #0c1a10; border: 1px solid #1a4020; border-left: 3px solid #3ddc84; border-radius: 0 8px 8px 0; padding: 10px 14px; font-size: 12px; color: #5aaa70; margin: 8px 0; }
.agent-box { background: #0a1020; border: 1px solid #1a3050; border-left: 3px solid #4a9eff; border-radius: 0 8px 8px 0; padding: 10px 14px; font-size: 11px; color: #4a7099; font-family: 'IBM Plex Mono', monospace; margin: 6px 0; }
.voice-active { background: #1a0808; border: 1px solid #4a1010; border-radius: 8px; padding: 8px 12px; margin: 4px 0; font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #cc4444; }
.metric-card { background: #080d18; border: 1px solid #0f1e38; border-radius: 8px; padding: 12px 16px; text-align: center; }
.metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 22px; font-weight: 600; color: #4a9eff; }
.metric-label { font-size: 10px; color: #445566; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 2px; }
section[data-testid="stSidebar"] { background: #060b16 !important; border-right: 1px solid #0f1e38; }
hr { border-color: #0f1e38 !important; }
textarea[data-testid="stChatInputTextArea"] { background: #0a1020 !important; color: #c8d0e0 !important; border: 1px solid #1e3050 !important; font-family: 'IBM Plex Sans', sans-serif !important; font-size: 14px !important; }
</style>
""", unsafe_allow_html=True)


# ── LOAD PIPELINE ──
@st.cache_resource(show_spinner=False)
def load_pipeline():
    from retrieval.retrieval_pipeline import ForensicsRetriever
    from generation.generation import ForensicsGenerator
    from retrieval.langgraph_orchestrator import FFRAGAgent
    from ui.features import VoiceInput, Guardrails, GraphRenderer, ResponseFormatter
    return (
        ForensicsRetriever(),
        ForensicsGenerator(),
        FFRAGAgent(),
        VoiceInput(),
        Guardrails,
        GraphRenderer,
        ResponseFormatter,
    )

with st.spinner("Loading forensics pipeline..."):
    try:
        retriever, generator, agent, voice, Guardrails, GraphRenderer, ResponseFormatter = load_pipeline()
    except Exception as e:
        st.error(f"Pipeline failed to load: {e}")
        st.stop()

if "messages"    not in st.session_state: st.session_state.messages    = []
if "session_id"  not in st.session_state: st.session_state.session_id  = str(uuid.uuid4())
if "session_name"not in st.session_state: st.session_state.session_name= "New Session"
if "load_sid"    not in st.session_state: st.session_state.load_sid    = None

# ── HEADER WITH METRICS (TOP RIGHT) ──
col_header_left, col_header_right = st.columns([3, 1])

with col_header_left:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:16px;padding:24px 0 8px;border-bottom:1px solid #1e2d4a;margin-bottom:24px;">
      <div>
        <div class="ffrag-logo">⬡ FFRAG</div>
        <div class="ffrag-subtitle">Financial Forensics · Agentic RAG · GraphRAG</div>
      </div>
    </div>""", unsafe_allow_html=True)

with col_header_right:
    # Load and display metrics in top right
    try:
        import json
        from pathlib import Path
        metrics_file = Path("evaluation/retrieval_metrics.json")
        if metrics_file.exists():
            metrics = json.loads(metrics_file.read_text())
            hybrid = metrics.get("hybrid", {})
            
            # Create a column for the metrics card
            if st.button("📊 Metrics", key="metrics_header_btn", help="Click to view full dashboard", use_container_width=False):
                st.session_state["show_full_dashboard"] = not st.session_state.get("show_full_dashboard", False)
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #0f1e38 0%, #1a2f52 100%);
                border: 1px solid #00d9ff;
                padding: 12px;
                border-radius: 6px;
                text-align: center;
                margin-top: 4px;
            ">
                <div style="color:#00d9ff;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Retrieval Metrics</div>
                <div style="color:#3ddc84;font-size:14px;font-weight:bold;margin-bottom:2px;">{hybrid.get('mean_precision', 0):.1%} Prec</div>
                <div style="color:#f0c040;font-size:12px;margin-bottom:2px;">MRR: {hybrid.get('mrr', 0):.3f}</div>
                <div style="color:#555;font-size:9px;margin-top:4px;">Recall: {hybrid.get('mean_recall', 0):.1%}</div>
            </div>
            """, unsafe_allow_html=True)
    except Exception:
        pass

# ── FULL DASHBOARD MODAL (TOP RIGHT) ──
if st.session_state.get("show_full_dashboard", False):
    st.markdown("""
    <hr style="border: 1px solid #1e3050; margin: 24px 0;">
    """, unsafe_allow_html=True)
    
    dashboard_col = st.container()
    with dashboard_col:
        if st.button("✕ Close Dashboard", key="close_dashboard_btn", use_container_width=False):
            st.session_state["show_full_dashboard"] = False
            st.rerun()
        try:
            from ui.metrics_dashboard import render_dashboard_content
            render_dashboard_content()
        except Exception as e:
            st.error(f"❌ Failed to load dashboard: {str(e)}")

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("### 🗂 Data Sources")
    c1, c2 = st.columns(2)
    c1.markdown('<div class="metric-card"><div class="metric-value">1,000</div><div class="metric-label">Transactions</div></div>', unsafe_allow_html=True)
    c2.markdown('<div class="metric-card"><div class="metric-value">9</div><div class="metric-label">Graph PNGs</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    c3.markdown('<div class="metric-card"><div class="metric-value">1,201</div><div class="metric-label">Reg Chunks</div></div>', unsafe_allow_html=True)
    c4.markdown('<div class="metric-card"><div class="metric-value">2,210</div><div class="metric-label">Total Docs</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🔍 Try These Queries")
    for s in ["Which accounts sent money to UAE?","Show structuring transactions below £10,000",
              "SAR filing timeline for continuing activity","Find dormant accounts suddenly reactivated",
              "What does FATF say about placement and aggregation?","Explain layering patterns","High risk corridors to Turkey or Morocco"]:
        if st.button(s, key=f"sug_{s}", use_container_width=True):
            st.session_state["pending_query"] = s
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    top_k        = st.slider("Results per query", 3, 10, 5)
    show_scores  = st.toggle("Show rerank scores", value=False)
    show_sources = st.toggle("Show source chunks", value=True)
    show_graphs  = st.toggle("Render graph images", value=True)
    show_agent   = st.toggle("Show agent trace", value=True)
    voice_mode   = st.toggle("🎙 Voice input", value=False)
    
    st.markdown("---")
    st.markdown('<div style="font-size:11px;color:#334455;line-height:1.6;"><b style="color:#445566">Pipeline:</b> LangGraph Agentic RAG<br><b style="color:#445566">Retrieval:</b> BM25 + Dense + Reranker<br><b style="color:#445566">Graph DB:</b> Neo4j AuraDB<br><b style="color:#445566">Generation:</b> Llama 3.3 70B via Groq<br><b style="color:#445566">Voice:</b> Whisper Large v3 via Groq</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💾 Sessions")

    # Save current session button
    col_s, col_n = st.columns([1,1])
    with col_s:
        if st.button("💾 Save", use_container_width=True):
            if st.session_state.messages:
                st.session_state.session_name = auto_name(st.session_state.messages)
                save_session(st.session_state.session_id, st.session_state.session_name, st.session_state.messages)
                st.success("Saved!")
    with col_n:
        if st.button("➕ New", use_container_width=True):
            if st.session_state.messages:
                save_session(st.session_state.session_id, auto_name(st.session_state.messages), st.session_state.messages)
            st.session_state.messages    = []
            st.session_state.session_id  = str(uuid.uuid4())
            st.session_state.session_name= "New Session"
            st.rerun()

    # Previous sessions list
    prev_sessions = load_all_sessions()
    if prev_sessions:
        st.markdown('<div style="font-size:10px;color:#334455;letter-spacing:1px;text-transform:uppercase;margin:8px 0 4px;">Previous Sessions</div>', unsafe_allow_html=True)
        for s in prev_sessions[:10]:
            is_current = s["id"] == st.session_state.session_id
            label = f"{'▶ ' if is_current else ''}{s['name']}"
            col_l, col_d = st.columns([5,1])
            with col_l:
                if st.button(label, key=f"load_{s['id']}", use_container_width=True, disabled=is_current):
                    if st.session_state.messages:
                        save_session(st.session_state.session_id, auto_name(st.session_state.messages), st.session_state.messages)
                    loaded = load_session(s["id"])
                    st.session_state.messages     = loaded
                    st.session_state.session_id   = s["id"]
                    st.session_state.session_name = s["name"]
                    st.rerun()
            with col_d:
                if st.button("🗑", key=f"del_{s['id']}", use_container_width=True):
                    delete_session(s["id"])
                    if s["id"] == st.session_state.session_id:
                        st.session_state.messages    = []
                        st.session_state.session_id  = str(uuid.uuid4())
                        st.session_state.session_name= "New Session"
                    st.rerun()
    else:
        st.caption("No saved sessions yet.")

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── HELPERS ──
def _get_doc_pages(filename):
    return {"bulletin-2025-31a.pdf":4,"FATF Recommendations 2012.pdf":152,
            "Guidance-PEP-Rec12-22.pdf":36,"Professional-Money-Laundering.pdf":53,
            "sar_tti_01.pdf":33}.get(filename, 50)

def humanise_citations(text, results):
    maps = {}
    for i, r in enumerate(results.get("transactions",[]), 1):
        m = r["metadata"]
        maps[f"[TXN-{i}]"] = f"**Account {m.get('sender_account','?')}→{m.get('receiver_account','?')} (£{float(m.get('amount',0)):,.0f}, {m.get('sender_location','?')}→{m.get('receiver_location','?')}, {m.get('typology','?')})**"
    for i, r in enumerate(results.get("graph_captions",[]), 1):
        m = r["metadata"]
        maps[f"[GRAPH-{i}]"] = f"**{m.get('title','Graph')} — {m.get('n_accounts','?')} accounts, £{float(m.get('total_volume',0)):,.0f} total**"
    for i, r in enumerate(results.get("regulations",[]), 1):
        m = r["metadata"]
        fname = m.get("filename","Regulation")
        readable = fname.replace(".pdf","").replace("-"," ").replace("_"," ").strip()
        name_map = {
            "bulletin 2025 31a":             "OCC/FinCEN SAR FAQ (2025)",
            "FATF Recommendations 2012":     "FATF Recommendations (2012)",
            "Guidance PEP Rec12 22":         "FATF PEP Guidance (Rec.12 & 22)",
            "Professional Money Laundering": "FATF Professional Money Laundering Report",
            "sar tti 01":                    "FinCEN SAR Activity Review",
        }
        friendly = name_map.get(readable) or next((v for k,v in name_map.items() if k.lower() in readable.lower()), readable)
        ci = m.get("chunk_idx",""); nc = m.get("n_chunks",1)
        page = f"p.~{max(1,round((int(ci)/int(nc))*_get_doc_pages(fname)))}" if str(ci).isdigit() and nc else ""
        maps[f"[REG-{i}]"] = f"**{friendly}{' — '+page if page else ''}**"
    for tag, label in maps.items():
        text = text.replace(tag, label)
    return text

def render_badges(sources):
    if not sources: return ""
    h = '<div class="source-badges">'
    if "transactions"   in sources: h += '<span class="badge badge-txn">📊 Transactions</span>'
    if "graph_captions" in sources: h += '<span class="badge badge-graph">🕸 Graph Analysis</span>'
    if "regulations"    in sources: h += '<span class="badge badge-reg">📄 Regulations</span>'
    return h + '</div>'

def render_score(sd):
    if not sd: return ""
    cls = {"CRITICAL":"score-critical","HIGH":"score-high","MEDIUM":"score-medium","LOW":"score-low"}.get(sd.get("level",""),"score-medium")
    flags = " · ".join(sd.get("flags",[]))
    return f'<div class="score-block {cls}">🚨 {sd.get("score",0)}/10 [{sd.get("level","")}] &nbsp;|&nbsp; {sd.get("reason","")}{"&nbsp;·&nbsp;"+flags if flags else ""}</div>'

def render_agent_trace(output):
    """Show agentic loop metadata — pipeline chosen, retries, relevance score."""
    pipeline  = output.get("pipeline", "both")
    retries   = output.get("retry_count", 0)
    relevance = output.get("relevance_score", 0.0)
    retry_str = f" · 🔄 self-corrected {retries}x" if retries > 0 else ""
    return (
        f'<div class="agent-box">'
        f'⬡ AGENT · pipeline={pipeline} · relevance={relevance:.2f}{retry_str}'
        f'</div>'
    )

def render_expander(results, show_scores):
    tabs_l = []
    if results.get("transactions"):   tabs_l.append(f"📊 TXN ({len(results['transactions'])})")
    if results.get("graph_captions"): tabs_l.append(f"🕸 Graphs ({len(results['graph_captions'])})")
    if results.get("regulations"):    tabs_l.append(f"📄 Regs ({len(results['regulations'])})")
    if not tabs_l: return
    tabs = st.tabs(tabs_l); idx = 0
    if results.get("transactions"):
        with tabs[idx]:
            for r in results["transactions"]:
                m = r["metadata"]
                susp_label  = '🚨 **SUSPICIOUS**' if m.get('is_suspicious') else '✅ Normal'
                score_label = f' | Rerank: `{r.get("rerank_score", 0):.3f}`' if show_scores else ''
                st.markdown(
                    f"**Account:** `{m.get('sender_account','?')}` → `{m.get('receiver_account','?')}`\n\n"
                    f"**Amount:** £{float(m.get('amount',0)):,.2f} | **Typology:** `{m.get('typology','?')}`\n\n"
                    f"**Route:** {m.get('sender_location','?')} → {m.get('receiver_location','?')}\n\n"
                    f"{susp_label}{score_label}"
                )
                st.divider()
        idx += 1
    if results.get("graph_captions"):
        with tabs[idx]:
            for r in results["graph_captions"]:
                st.markdown(f"**Graph:** `{r['metadata'].get('graph_id','?')}`")
                st.markdown(r["document"][:600]+"...")
                if show_scores: st.caption(f"Rerank: {r.get('rerank_score',0):.3f}")
                st.divider()
        idx += 1
    if results.get("regulations"):
        with tabs[idx]:
            for r in results["regulations"]:
                st.markdown(f"**Source:** `{r['metadata'].get('filename','?')}` — chunk {r['metadata'].get('chunk_idx','?')}")
                st.markdown(r["document"][:400]+"...")
                if show_scores: st.caption(f"Rerank: {r.get('rerank_score',0):.3f}")
                st.divider()


# ── WELCOME ──
if not st.session_state.messages:
    st.markdown(
        '<div class="assistant-bubble">'
        '<div style="color:#4a9eff;font-family:\'IBM Plex Mono\',monospace;font-size:12px;letter-spacing:2px;margin-bottom:10px;">FFRAG ONLINE · AGENTIC MODE</div>'
        '<div style="font-size:14px;color:#8090a8;line-height:1.8;">'
        'I\'m your Financial Forensics AI running a <b style="color:#4a9eff">LangGraph agentic loop</b> — '
        'I route your query, expand it, retrieve evidence, grade my own results, and self-correct if needed.<br><br>'
        'I reason across <b style="color:#3ddc84">transaction records</b>, '
        '<b style="color:#f0c040">wallet network graphs</b>, and '
        '<b style="color:#c084f0">regulatory documents</b> simultaneously.'
        '</div></div>',
        unsafe_allow_html=True
    )

# ── RENDER HISTORY ──
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">🔎 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        html = (
            f'<div class="assistant-bubble">{msg["content"]}'
            + render_badges(msg.get("sources",[]))
            + render_score(msg.get("score_data"))
            + "</div>"
        )
        st.markdown(html, unsafe_allow_html=True)
        if show_agent and msg.get("agent_output"):
            st.markdown(render_agent_trace(msg["agent_output"]), unsafe_allow_html=True)
        if show_graphs and msg.get("retrieval"):
            GraphRenderer.render(msg["retrieval"])
        if show_sources and msg.get("retrieval"):
            with st.expander("📂 Retrieved Context", expanded=False):
                render_expander(msg["retrieval"], show_scores)


# ── QUERY HANDLER ──
def handle_query(query):
    guard = Guardrails.check_input(query)
    if not guard["allowed"]:
        st.markdown(f'<div class="guardrail-block">{guard["response"]}</div>', unsafe_allow_html=True)
        st.session_state.messages += [
            {"role":"user","content":query},
            {"role":"assistant","content":guard["response"],"sources":[],"score_data":None,"retrieval":{},"agent_output":None}
        ]
        return

    st.markdown(f'<div class="user-bubble">🔎 {query}</div>', unsafe_allow_html=True)
    if guard.get("warning"):
        st.markdown(f'<div class="warning-block">💡 {guard["warning"]}</div>', unsafe_allow_html=True)
    if any(w in query.lower() for w in ["smurf","aggregat"]):
        st.markdown('<div class="tip-box">💡 <b>Terminology note:</b> FATF formally calls smurfing <b>"placement via aggregation"</b>. Query expanded automatically.</div>', unsafe_allow_html=True)

    # ── Agentic retrieval + generation ──
    with st.spinner("⬡ Agent reasoning — routing → expanding → retrieving → grading..."):
        output = agent.run(query)

    results    = output["raw_results"]
    raw        = output["answer"]
    sources    = output["sources"]
    contexts   = [r["document"] for r in results.get("all_results", [])]
    out_guard  = Guardrails.check_output(raw, contexts)
    if not out_guard["valid"]:
        raw = f"⚠️ {out_guard['issue']} Please try rephrasing."
    answer     = ResponseFormatter.format(humanise_citations(raw, results))
    score_data = output.get("suspicion_score")

    # ── Render answer ──
    html = (
        f'<div class="assistant-bubble">{answer}'
        + render_badges(sources)
        + render_score(score_data)
        + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)

    # ── Agent trace ──
    if show_agent:
        st.markdown(render_agent_trace(output), unsafe_allow_html=True)

    # ── Graph images ──
    if show_graphs:
        GraphRenderer.render(results)

    # ── Source chunks ──
    if show_sources:
        with st.expander("📂 Retrieved Context", expanded=False):
            render_expander(results, show_scores)

    # ── Save to history ──
    st.session_state.messages += [
        {"role":"user","content":query},
        {
            "role":         "assistant",
            "content":      answer,
            "sources":      sources,
            "score_data":   score_data,
            "retrieval":    results,
            "agent_output": output,
        }
    ]
    # Auto-save session after every exchange
    st.session_state.session_name = auto_name(st.session_state.messages)
    save_session(st.session_state.session_id, st.session_state.session_name, st.session_state.messages)


# ── VOICE INPUT ──
if voice_mode:
    st.markdown("---")
    try:
        audio_bytes = voice.render_widget()
        if audio_bytes and len(audio_bytes) > 1000:
            if st.button("📤 Transcribe & Send", use_container_width=True):
                with st.spinner("🎙 Transcribing..."):
                    try:
                        transcript = voice.transcribe(audio_bytes)
                        st.markdown(f'<div class="voice-active">🎙 Transcribed: "{transcript}"</div>', unsafe_allow_html=True)
                        handle_query(transcript)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")
    except Exception as e:
        st.error(f"Voice input error: {e}")


# ── INPUT ──
if "pending_query" in st.session_state:
    handle_query(st.session_state.pop("pending_query"))
    st.rerun()

if prompt := st.chat_input("Ask about transactions, typologies, regulations..."):
    handle_query(prompt)
    st.rerun()
