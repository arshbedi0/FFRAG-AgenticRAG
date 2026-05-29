"""
app.py — Financial Forensics RAG Chatbot UI (Streamlit Cloud Ready)
Run: streamlit run ui/app.py
"""

import os, sys, random, re, json, uuid, datetime, subprocess
from pathlib import Path
sys.path.append(".")

import streamlit as st
import streamlit.components.v1 as components
from ui.motion import (
    MOTION_CSS, BACKGROUND_CANVAS,
    animated_metric_card, get_ai_thinking_html, get_retrieval_timeline,
    get_fire_trigger_js,
)

st.set_page_config(
    page_title="FFRAG — Financial Forensics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

def _load_secrets():
    try:
        for k in ["GROQ_API_KEY", "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD",
                  "EMBEDDING_MODEL", "LLM_MODEL", "CHROMA_DIR"]:
            if k in st.secrets and not os.environ.get(k):
                os.environ[k] = st.secrets[k]
    except Exception:
        pass
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

_load_secrets()

def _ensure_chroma():
    sqlite = Path("chroma_db/chroma.sqlite3")
    if sqlite.exists():
        return True, None
    try:
        r = subprocess.run(
            [sys.executable, "ingestion/ingest_to_chroma.py"],
            capture_output=True, text=True, timeout=300
        )
        if r.returncode == 0 and sqlite.exists():
            return True, None
        return False, r.stderr[:300]
    except Exception as e:
        return False, str(e)

_chroma_ok, _chroma_err = _ensure_chroma()
if not _chroma_ok:
    st.error("🚨 chroma_db/chroma.sqlite3 not found.")
    st.info("Make sure `chroma_db/` is committed to your GitHub repo and not in `.gitignore`.")
    if _chroma_err:
        st.code(_chroma_err)
    st.stop()

@st.cache_resource(show_spinner=False)
def load_pipeline():
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not set. Add it in Streamlit Cloud → Settings → Secrets.")

    from ui.features import Guardrails, GraphRenderer, ResponseFormatter
    from retrieval.retrieval_pipeline import ForensicsRetriever
    retriever = ForensicsRetriever()
    from generation.generation import ForensicsGenerator
    generator = ForensicsGenerator()

    agent = None
    try:
        from retrieval.langgraph_orchestrator import FFRAGAgent
        agent = FFRAGAgent(retriever=retriever)
    except Exception:
        pass

    voice = None
    try:
        from ui.features import VoiceInput
        voice = VoiceInput()
    except Exception:
        pass

    return retriever, generator, agent, voice, Guardrails, GraphRenderer, ResponseFormatter

with st.spinner("⬡ Loading forensics pipeline..."):
    try:
        retriever, generator, agent, voice, Guardrails, GraphRenderer, ResponseFormatter = load_pipeline()
    except RuntimeError as e:
        st.error(str(e))
        st.info("Add your secrets in **Streamlit Cloud → App settings → Secrets** (TOML format):")
        st.code('GROQ_API_KEY = "gsk_..."\nNEO4J_URI = "neo4j+s://xxxx.databases.neo4j.io"\nNEO4J_USER = "neo4j"\nNEO4J_PASSWORD = "your-password"', language="toml")
        st.stop()
    except Exception as e:
        st.error(f"Startup error: {e}")
        st.exception(e)
        st.stop()

if "messages"          not in st.session_state: st.session_state.messages          = []
if "session_id"        not in st.session_state: st.session_state.session_id        = str(uuid.uuid4())
if "session_name"      not in st.session_state: st.session_state.session_name      = "New Session"
if "last_audio_hash"   not in st.session_state: st.session_state.last_audio_hash   = None
if "voice_transcript"  not in st.session_state: st.session_state.voice_transcript  = ""
if "voice_ready"       not in st.session_state: st.session_state.voice_ready       = False

SESSIONS_DIR = Path("chat_sessions")
SESSIONS_DIR.mkdir(exist_ok=True)

def save_session(session_id, name, messages):
    path = SESSIONS_DIR / f"{session_id}.json"
    path.write_text(json.dumps({
        "id": session_id, "name": name,
        "created": str(datetime.datetime.now()),
        "messages": [{k: v for k, v in m.items() if k != "retrieval"} for m in messages],
    }, indent=2, default=str))

def load_all_sessions():
    out = []
    for p in sorted(SESSIONS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            d = json.loads(p.read_text())
            preview = next((m["content"][:50] for m in d.get("messages", []) if m["role"] == "user"), "Empty")
            out.append({"id": d["id"], "name": d["name"], "created": d["created"], "preview": preview})
        except Exception:
            pass
    return out

def load_session(session_id):
    p = SESSIONS_DIR / f"{session_id}.json"
    return json.loads(p.read_text()).get("messages", []) if p.exists() else []

def delete_session(session_id):
    p = SESSIONS_DIR / f"{session_id}.json"
    if p.exists(): p.unlink()

def auto_name(messages):
    for m in messages:
        if m["role"] == "user":
            q = m["content"]
            return q[:40] + ("..." if len(q) > 40 else "")
    return "New Session"

# ── MOTION SYSTEM ──
st.markdown(MOTION_CSS, unsafe_allow_html=True)
st.markdown(BACKGROUND_CANVAS, unsafe_allow_html=True)


# ── HEADER ──
col_left, col_right = st.columns([3, 1])
with col_left:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:16px;padding:20px 0 8px;
                border-bottom:1px solid #0f1e38;margin-bottom:20px;">
      <div>
        <div class="ffrag-logo">⬡ FFRAG</div>
        <div class="ffrag-subtitle">Financial Forensics · Agentic RAG · GraphRAG</div>
      </div>
    </div>""", unsafe_allow_html=True)

with col_right:
    try:
        metrics_file = Path("evaluation/retrieval_metrics.json")
        if metrics_file.exists():
            hybrid = json.loads(metrics_file.read_text()).get("hybrid", {})
            if st.button("📊 Metrics", key="metrics_btn"):
                st.session_state["show_dashboard"] = not st.session_state.get("show_dashboard", False)
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#0d1e38,#162f52);border:1px solid #1a4070;
                        padding:12px;border-radius:8px;text-align:center;margin-top:4px;">
                <div style="color:#4a9eff;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Retrieval Metrics</div>
                <div style="color:#3ddc84;font-size:14px;font-weight:bold;">{hybrid.get('mean_precision',0):.1%} Prec</div>
                <div style="color:#f0c040;font-size:12px;">MRR: {hybrid.get('mrr',0):.3f}</div>
                <div style="color:#445566;font-size:9px;margin-top:4px;">Recall: {hybrid.get('mean_recall',0):.1%}</div>
            </div>""", unsafe_allow_html=True)
    except Exception:
        pass

if st.session_state.get("show_dashboard"):
    st.markdown("---")
    if st.button("✕ Close", key="close_dash"):
        st.session_state["show_dashboard"] = False
        st.rerun()
    try:
        from ui.metrics_dashboard import render_dashboard_content
        render_dashboard_content()
    except Exception as e:
        st.error(f"Dashboard unavailable: {e}")

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("### 🗂 Data Sources")
    c1, c2 = st.columns(2)
    c1.markdown(animated_metric_card("Transactions", "1000",  nth=0), unsafe_allow_html=True)
    c2.markdown(animated_metric_card("Graph PNGs",   "9",     nth=1), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    c3.markdown(animated_metric_card("Reg Chunks",   "1201",  nth=2), unsafe_allow_html=True)
    c4.markdown(animated_metric_card("Total Docs",   "2210",  nth=3), unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🔍 Try These Queries")
    for s in ["Which accounts sent money to UAE?", "Show structuring transactions below £10,000",
              "SAR filing timeline for continuing activity", "Find dormant accounts suddenly reactivated",
              "What does FATF say about placement and aggregation?", "Explain layering patterns",
              "High risk corridors to Turkey or Morocco"]:
        if st.button(s, key=f"sug_{s}", use_container_width=True):
            st.session_state["pending_query"] = s
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    top_k        = st.slider("Results per query", 3, 10, 5)
    show_scores  = st.toggle("Show rerank scores",  value=False)
    show_sources = st.toggle("Show source chunks",  value=True)
    show_graphs  = st.toggle("Render graph images", value=True)
    show_agent   = st.toggle("Show agent trace",    value=True)
    voice_mode   = st.toggle("🎙 Voice input",       value=False)
    st.markdown("---")
    st.markdown('<div style="font-size:11px;color:#334455;line-height:1.8;"><b style="color:#445566">Pipeline:</b> LangGraph Agentic RAG<br><b style="color:#445566">Retrieval:</b> BM25 + Dense + Reranker<br><b style="color:#445566">Graph DB:</b> Neo4j AuraDB<br><b style="color:#445566">Generation:</b> Llama 3.3 70B via Groq<br><b style="color:#445566">Voice:</b> Whisper Large v3 via Groq</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 💾 Sessions")
    cs, cn = st.columns(2)
    with cs:
        if st.button("💾 Save", use_container_width=True):
            if st.session_state.messages:
                st.session_state.session_name = auto_name(st.session_state.messages)
                save_session(st.session_state.session_id, st.session_state.session_name, st.session_state.messages)
                st.success("Saved!")
    with cn:
        if st.button("➕ New", use_container_width=True):
            if st.session_state.messages:
                save_session(st.session_state.session_id, auto_name(st.session_state.messages), st.session_state.messages)
            st.session_state.messages     = []
            st.session_state.session_id   = str(uuid.uuid4())
            st.session_state.session_name = "New Session"
            st.rerun()
    prev = load_all_sessions()
    if prev:
        st.markdown('<div style="font-size:10px;color:#334455;letter-spacing:1px;text-transform:uppercase;margin:8px 0 4px;">Previous Sessions</div>', unsafe_allow_html=True)
        for s in prev[:10]:
            is_cur = s["id"] == st.session_state.session_id
            cl, cd = st.columns([5, 1])
            with cl:
                if st.button(f"{'▶ ' if is_cur else ''}{s['name']}", key=f"load_{s['id']}", use_container_width=True, disabled=is_cur):
                    if st.session_state.messages:
                        save_session(st.session_state.session_id, auto_name(st.session_state.messages), st.session_state.messages)
                    st.session_state.messages     = load_session(s["id"])
                    st.session_state.session_id   = s["id"]
                    st.session_state.session_name = s["name"]
                    st.rerun()
            with cd:
                if st.button("🗑", key=f"del_{s['id']}", use_container_width=True):
                    delete_session(s["id"])
                    if s["id"] == st.session_state.session_id:
                        st.session_state.messages     = []
                        st.session_state.session_id   = str(uuid.uuid4())
                        st.session_state.session_name = "New Session"
                    st.rerun()
    else:
        st.caption("No saved sessions yet.")
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── HELPERS ──
def _get_doc_pages(filename):
    return {"bulletin-2025-31a.pdf": 4, "FATF Recommendations 2012.pdf": 152,
            "Guidance-PEP-Rec12-22.pdf": 36, "Professional-Money-Laundering.pdf": 53,
            "sar_tti_01.pdf": 33}.get(filename, 50)

def humanise_citations(text, results):
    maps = {}
    for i, r in enumerate(results.get("transactions", []), 1):
        m = r["metadata"]
        maps[f"[TXN-{i}]"] = f"**Account {m.get('sender_account','?')}→{m.get('receiver_account','?')} (£{float(m.get('amount',0)):,.0f}, {m.get('sender_location','?')}→{m.get('receiver_location','?')}, {m.get('typology','?')})**"
    for i, r in enumerate(results.get("graph_captions", []), 1):
        m = r["metadata"]
        maps[f"[GRAPH-{i}]"] = f"**{m.get('title','Graph')} — {m.get('n_accounts','?')} accounts, £{float(m.get('total_volume',0)):,.0f} total**"
    for i, r in enumerate(results.get("regulations", []), 1):
        m        = r["metadata"]
        fname    = m.get("filename", "Regulation")
        readable = fname.replace(".pdf", "").replace("-", " ").replace("_", " ").strip()
        name_map = {
            "bulletin 2025 31a":             "OCC/FinCEN SAR FAQ (2025)",
            "FATF Recommendations 2012":     "FATF Recommendations (2012)",
            "Guidance PEP Rec12 22":         "FATF PEP Guidance (Rec.12 & 22)",
            "Professional Money Laundering": "FATF Professional Money Laundering Report",
            "sar tti 01":                    "FinCEN SAR Activity Review",
        }
        friendly = name_map.get(readable) or next((v for k, v in name_map.items() if k.lower() in readable.lower()), readable)
        ci, nc   = m.get("chunk_idx", ""), m.get("n_chunks", 1)
        page     = f"p.~{max(1,round((int(ci)/int(nc))*_get_doc_pages(fname)))}" if str(ci).isdigit() and nc else ""
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
    return h + "</div>"

def render_score(sd):
    if not sd: return ""
    cls   = {"CRITICAL": "score-critical", "HIGH": "score-high", "MEDIUM": "score-medium", "LOW": "score-low"}.get(sd.get("level", ""), "score-medium")
    flags = " · ".join(sd.get("flags", []))
    return f'<div class="score-block {cls}">🚨 {sd.get("score",0)}/10 [{sd.get("level","")}] &nbsp;|&nbsp; {sd.get("reason","")}{"&nbsp;·&nbsp;"+flags if flags else ""}</div>'

def render_agent_trace(output):
    pipeline  = output.get("pipeline", "both")
    retries   = output.get("retry_count", 0)
    relevance = output.get("relevance_score", 0.0)
    retry_str = f" · 🔄 self-corrected {retries}x" if retries > 0 else ""
    return f'<div class="agent-box">⬡ AGENT · pipeline={pipeline} · relevance={relevance:.2f}{retry_str}</div>'

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
                st.markdown(
                    f"**Account:** `{m.get('sender_account','?')}` → `{m.get('receiver_account','?')}`\n\n"
                    f"**Amount:** £{float(m.get('amount',0)):,.2f} | **Typology:** `{m.get('typology','?')}`\n\n"
                    f"**Route:** {m.get('sender_location','?')} → {m.get('receiver_location','?')}\n\n"
                    f"{'🚨 **SUSPICIOUS**' if m.get('is_suspicious') else '✅ Normal'}"
                    f"{f' | Rerank: `{r.get(chr(114)+chr(101)+chr(114)+chr(97)+chr(110)+chr(107)+chr(95)+chr(115)+chr(99)+chr(111)+chr(114)+chr(101),0):.3f}`' if show_scores else ''}"
                )
                st.divider()
        idx += 1
    if results.get("graph_captions"):
        with tabs[idx]:
            for r in results["graph_captions"]:
                st.markdown(f"**Graph:** `{r['metadata'].get('graph_id','?')}`")
                st.markdown(r["document"][:600] + "...")
                if show_scores: st.caption(f"Rerank: {r.get('rerank_score',0):.3f}")
                st.divider()
        idx += 1
    if results.get("regulations"):
        with tabs[idx]:
            for r in results["regulations"]:
                st.markdown(f"**Source:** `{r['metadata'].get('filename','?')}` — chunk {r['metadata'].get('chunk_idx','?')}")
                st.markdown(r["document"][:400] + "...")
                if show_scores: st.caption(f"Rerank: {r.get('rerank_score',0):.3f}")
                st.divider()


# ── WELCOME ──
if not st.session_state.messages:
    st.markdown(
        '<div class="assistant-bubble">'
        '<div style="color:#4a9eff;font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
        'letter-spacing:3px;margin-bottom:12px;">FFRAG ONLINE · AGENTIC MODE</div>'
        '<div style="font-size:14px;color:#7a90a8;line-height:1.9;">'
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
        st.markdown(
            f'<div class="assistant-bubble">{msg["content"]}'
            + render_badges(msg.get("sources", []))
            + render_score(msg.get("score_data"))
            + "</div>", unsafe_allow_html=True
        )
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
            {"role": "user", "content": query},
            {"role": "assistant", "content": guard["response"], "sources": [],
             "score_data": None, "retrieval": {}, "agent_output": None},
        ]
        return

    components.html(get_fire_trigger_js(), height=0)
    st.markdown(f'<div class="user-bubble">🔎 {query}</div>', unsafe_allow_html=True)
    if guard.get("warning"):
        st.markdown(f'<div class="warning-block">💡 {guard["warning"]}</div>', unsafe_allow_html=True)
    if any(w in query.lower() for w in ["smurf", "aggregat"]):
        st.markdown('<div class="tip-box">💡 <b>Terminology note:</b> FATF formally calls smurfing <b>"placement via aggregation"</b>. Query expanded automatically.</div>', unsafe_allow_html=True)

    _loader = st.empty()

    if agent is not None:
        _loader.markdown(
            get_retrieval_timeline(active=0)
            + get_ai_thinking_html("Routing query → expanding → retrieving → grading..."),
            unsafe_allow_html=True,
        )
        output = agent.run(query)
        _loader.empty()
    else:
        _loader.markdown(
            get_retrieval_timeline(active=2)
            + get_ai_thinking_html("Retrieving evidence and generating response..."),
            unsafe_allow_html=True,
        )
        results  = retriever.retrieve(query, top_k=top_k)
        raw_out  = generator.generate(query, results)
        _loader.empty()
        output   = {
            "answer":          raw_out["answer"],
            "sources":         raw_out["sources"],
            "suspicion_score": generator.score_suspicion(results["transactions"][0]["document"]) if results.get("transactions") else None,
            "raw_results":     results,
            "relevance_score": 0.0,
            "retry_count":     0,
            "pipeline":        "direct",
        }

    results   = output["raw_results"]
    raw       = output["answer"]
    sources   = output["sources"]
    contexts  = [r["document"] for r in results.get("all_results", [])]
    out_guard = Guardrails.check_output(raw, contexts)
    if not out_guard["valid"]:
        raw = f"⚠️ {out_guard['issue']} Please try rephrasing."
    answer     = ResponseFormatter.format(humanise_citations(raw, results))
    score_data = output.get("suspicion_score")

    st.markdown(
        f'<div class="assistant-bubble">{answer}'
        + render_badges(sources)
        + render_score(score_data)
        + "</div>", unsafe_allow_html=True
    )
    if show_agent:
        st.markdown(render_agent_trace(output), unsafe_allow_html=True)
    if show_graphs:
        GraphRenderer.render(results)
    if show_sources:
        with st.expander("📂 Retrieved Context", expanded=False):
            render_expander(results, show_scores)

    st.session_state.messages += [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer, "sources": sources,
         "score_data": score_data, "retrieval": results, "agent_output": output},
    ]
    st.session_state.session_name = auto_name(st.session_state.messages)
    save_session(st.session_state.session_id, st.session_state.session_name, st.session_state.messages)


# ══════════════════════════════════════════════════════════════
# ── VOICE INPUT PANEL
# ══════════════════════════════════════════════════════════════
if voice_mode and voice is not None:
    try:
        from streamlit_mic_recorder import mic_recorder

        # ── STEP 1: Handle send/clear FIRST — before any audio/widget logic.
        # When the button is clicked, Streamlit reruns. On that rerun mic_recorder
        # returns None (just_once=True), so we must act on session_state alone.
        if st.session_state.get("voice_ready") and st.session_state.get("voice_transcript"):

            st.markdown("""
            <div class="voice-panel">
              <div class="voice-panel-header">
                <span class="voice-dot voice-dot-live"></span>
                <span class="voice-panel-title">Voice Input — Whisper Large v3</span>
              </div>
              <div class="voice-panel-body">
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="voice-transcript-box">
              <div class="voice-transcript-label">Transcribed</div>
              {st.session_state.voice_transcript}
            </div>
            """, unsafe_allow_html=True)

            col_send, col_clear = st.columns([3, 1])
            with col_send:
                send_clicked = st.button(
                    "📤  Send to FFRAG",
                    key="voice_send_btn",
                    use_container_width=True,
                    type="primary",
                )
            with col_clear:
                clear_clicked = st.button(
                    "✕ Clear",
                    key="voice_clear_btn",
                    use_container_width=True,
                )

            st.markdown("</div></div>", unsafe_allow_html=True)

            if clear_clicked:
                st.session_state.voice_transcript = ""
                st.session_state.voice_ready      = False
                st.session_state.last_audio_hash  = None
                st.rerun()

            if send_clicked:
                q = st.session_state.voice_transcript
                st.session_state.voice_transcript = ""
                st.session_state.voice_ready      = False
                st.session_state.last_audio_hash  = None
                st.session_state.pending_query    = q
                st.rerun()

        else:
            # ── STEP 2: No transcript yet — show recorder ──
            st.markdown("""
            <div class="voice-panel">
              <div class="voice-panel-header">
                <span class="voice-dot"></span>
                <span class="voice-panel-title">Voice Input — Whisper Large v3</span>
              </div>
              <div class="voice-panel-body">
            """, unsafe_allow_html=True)

            audio = mic_recorder(
                start_prompt="⏺  Start recording",
                stop_prompt="⏹  Stop recording",
                just_once=True,
                use_container_width=True,
                key="mic_recorder_widget",
            )

            if audio and audio.get("bytes") and len(audio["bytes"]) > 1000:
                audio_hash = hash(audio["bytes"])
                if audio_hash != st.session_state.last_audio_hash:
                    st.session_state.last_audio_hash  = audio_hash
                    st.session_state.voice_ready      = False
                    st.session_state.voice_transcript = ""
                    with st.spinner("🎙 Transcribing with Whisper..."):
                        try:
                            transcript = voice.transcribe(audio["bytes"], filename="audio.wav")
                            st.session_state.voice_transcript = transcript.strip()
                            st.session_state.voice_ready      = True
                            st.rerun()  # rerun to show transcript + send button
                        except Exception as e:
                            st.markdown(
                                f'<div class="voice-error">⚠ Transcription failed: {e}</div>',
                                unsafe_allow_html=True
                            )
            else:
                st.markdown(
                    '<div class="voice-send-hint">Press ⏺ to record · auto-transcribes on stop</div>',
                    unsafe_allow_html=True
                )

            st.markdown("</div></div>", unsafe_allow_html=True)

    except ImportError:
        st.markdown("""
        <div class="voice-panel">
          <div class="voice-panel-header">
            <span class="voice-dot"></span>
            <span class="voice-panel-title">Voice Input</span>
          </div>
          <div class="voice-panel-body">
            <div class="voice-error">
              ⚠ <b>streamlit-mic-recorder</b> not installed.<br>
              Add <code>streamlit-mic-recorder</code> to requirements.txt and redeploy.
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

elif voice_mode and voice is None:
    st.markdown("""
    <div class="voice-panel">
      <div class="voice-panel-header">
        <span class="voice-dot"></span>
        <span class="voice-panel-title">Voice Input</span>
      </div>
      <div class="voice-panel-body">
        <div class="voice-error">
          ⚠ VoiceInput failed to initialise — check GROQ_API_KEY is set in Secrets.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── PENDING SIDEBAR QUERY ──
if "pending_query" in st.session_state:
    handle_query(st.session_state.pop("pending_query"))
    st.rerun()

# ── TEXT INPUT ──
if prompt := st.chat_input("Ask about transactions, typologies, regulations..."):
    handle_query(prompt)
    st.rerun()
