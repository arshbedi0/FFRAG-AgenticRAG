"""
features.py
────────────
Drop-in feature additions for FFRAG app.py

Contains:
  1. VoiceInput        — Groq Whisper transcription
  2. GraphRenderer     — inline PNG rendering from metadata
  3. Guardrails        — input/output safety + topic enforcement
  4. ResponseFormatter — structured sections + varied phrasing

Import in app.py:
  from ui.features import VoiceInput, GraphRenderer, Guardrails, ResponseFormatter
"""

import os, re, random
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ══════════════════════════════════════════════════════════════
# 1. VOICE INPUT — Groq Whisper API
# ══════════════════════════════════════════════════════════════
class VoiceInput:
    """
    Records audio via streamlit-audiorec, transcribes via Groq Whisper.

    Install: pip install streamlit-audiorec groq
    """

    SUPPORTED_FORMATS = ["wav", "mp3", "mp4", "webm", "m4a", "ogg"]

    def __init__(self):
        from groq import Groq
        self.client = Groq(api_key=GROQ_API_KEY)

    def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        """Transcribe audio bytes to text via Groq Whisper."""
        import tempfile

        with tempfile.NamedTemporaryFile(
            suffix=f".{filename.split('.')[-1]}", delete=False
        ) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            with open(tmp_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-large-v3-turbo",  # fastest Groq Whisper
                    file=audio_file,
                    response_format="text",
                    language="en",
                )
            return transcription.strip()
        finally:
            os.unlink(tmp_path)

    @staticmethod
    def render_widget() -> bytes | None:
        """
        Renders the audio recorder widget in Streamlit.
        Returns audio bytes if recording available, else None.
        """
        try:
            from streamlit_audiorec import st_audiorec
            import streamlit as st

            st.markdown("""
            <div style="margin: 8px 0 4px;">
                <span style="font-family:'IBM Plex Mono',monospace; font-size:10px;
                             color:#445577; letter-spacing:2px;">
                    🎙 VOICE INPUT
                </span>
            </div>""", unsafe_allow_html=True)

            audio_bytes = st_audiorec()
            return audio_bytes

        except ImportError:
            import streamlit as st
            st.caption("Install streamlit-audiorec for voice input: "
                      "`pip install streamlit-audiorec`")
            return None


# ══════════════════════════════════════════════════════════════
# 2. GRAPH RENDERER — inline PNG from retrieval metadata
# ══════════════════════════════════════════════════════════════
class GraphRenderer:
    """
    Renders wallet network graph PNGs inline when graph captions are retrieved.
    Looks up image_path from caption metadata.
    """

    # Fallback path patterns to try if metadata path doesn't exist
    SEARCH_PATHS = [
        "{image_path}",
        "data/graphs/{graph_id}.png",
        "DATA/graphs/{graph_id}.png",
        "wallet_graphs/{graph_id}.png",
    ]

    @staticmethod
    def find_image(metadata: dict) -> str | None:
        """Find the graph PNG from metadata, trying multiple path patterns."""
        graph_id   = metadata.get("graph_id", "")
        image_path = metadata.get("image_path", "")

        for pattern in GraphRenderer.SEARCH_PATHS:
            path = pattern.format(
                image_path=image_path,
                graph_id=graph_id
            )
            if path and os.path.exists(path):
                return path
        return None

    @staticmethod
    def render(results: dict) -> None:
        """
        Renders all graph PNGs from retrieval results inline in Streamlit.
        Call this inside handle_query() after the answer is shown.
        """
        import streamlit as st

        graph_results = results.get("graph_captions", [])
        if not graph_results:
            return

        for r in graph_results:
            meta     = r["metadata"]
            title    = meta.get("title", "Graph Analysis")
            typology = meta.get("typology", "")
            vol      = float(meta.get("total_volume", 0))
            n_accs   = meta.get("n_accounts", "?")
            susp_vol = float(meta.get("suspicious_vol", 0))

            image_path = GraphRenderer.find_image(meta)

            if image_path:
                st.markdown(f"""
                <div style="
                    background:#080d18;
                    border:1px solid #1e3050;
                    border-left: 3px solid #f0c040;
                    border-radius: 0 8px 8px 0;
                    padding: 12px 16px;
                    margin: 8px 0 4px;
                ">
                    <span style="font-family:'IBM Plex Mono',monospace; font-size:10px;
                                 color:#f0c040; letter-spacing:2px;">
                        🕸 WALLET NETWORK GRAPH — {typology.upper()}
                    </span>
                    <div style="font-size:11px; color:#445577; margin-top:4px;">
                        {n_accs} accounts · £{vol:,.0f} total · 
                        £{susp_vol:,.0f} suspicious
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.image(
                    image_path,
                    caption=f"{title} — {n_accs} accounts, £{vol:,.0f} volume",
                    use_container_width=True,
                )
            else:
                st.warning(f"Graph image not found for {typology}. "
                          f"Expected at: {meta.get('image_path', 'unknown')}")


# ══════════════════════════════════════════════════════════════
# 3. GUARDRAILS
# ══════════════════════════════════════════════════════════════
class Guardrails:
    """
    Input + output guardrails for the FFRAG system.

    Input guardrails:
      - Block clearly off-topic queries
      - Detect prompt injection attempts
      - Warn on vague queries that will produce poor results

    Output guardrails:
      - Verify answer references retrieved context
      - Strip any leaked system prompt fragments
      - Flag low-confidence answers
    """

    # ── Topics that are in scope ──
    AML_KEYWORDS = [
        "transaction", "account", "suspicious", "aml", "sar", "laundering",
        "fatf", "fincen", "structuring", "smurfing", "layering", "corridor",
        "dormant", "currency", "round trip", "typology", "compliance", "bsa",
        "bank", "transfer", "payment", "fraud", "forensic", "investigation",
        "risk", "sanctions", "pep", "beneficial owner", "shell", "wire",
        "swift", "ach", "cross-border", "graph", "network", "pattern",
        "reactivat", "aggregat", "placement", "integration", "mule",
        "reporting", "threshold", "ctr", "kyc", "cdd", "edd",
        "uk", "uae", "turkey", "morocco", "nigeria", "india", "pakistan",
    ]

    # ── Prompt injection patterns ──
    INJECTION_PATTERNS = [
        r"ignore (previous|all|your) instructions",
        r"you are now",
        r"pretend (you are|to be)",
        r"forget (everything|all|your)",
        r"system prompt",
        r"jailbreak",
        r"dan mode",
        r"as an ai (without|with no) restrictions",
        r"disregard (your|all) (previous|prior)",
    ]

    # ── Soft redirect responses (varied) ──
    REDIRECT_RESPONSES = [
        "I'm a Financial Forensics specialist focused on AML investigation. "
        "I can help you analyse suspicious transactions, understand typologies, "
        "or query regulatory requirements. What would you like to investigate?",

        "That's outside my forensics scope. I'm built to reason across "
        "transaction records, wallet graphs, and AML regulations. "
        "Try asking about a specific account, typology, or SAR requirement.",

        "My expertise is financial crime investigation — structuring, layering, "
        "smurfing, high-risk corridors, and regulatory compliance. "
        "What aspect of the dataset would you like to explore?",
    ]

    @classmethod
    def check_input(cls, query: str) -> dict:
        """
        Returns:
          {"allowed": True}  — proceed normally
          {"allowed": False, "reason": "...", "response": "..."}  — block/redirect
          {"allowed": True,  "warning": "..."}  — allow but warn
        """
        q_lower = query.lower().strip()

        # 1. Empty query
        if len(q_lower) < 3:
            return {
                "allowed": False,
                "reason":  "empty_query",
                "response": "Please enter a question about transactions, "
                            "typologies, or AML regulations.",
            }

        # 2. Prompt injection
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, q_lower):
                return {
                    "allowed": False,
                    "reason":  "injection_attempt",
                    "response": "⚠️ That query pattern isn't supported. "
                                "I'm here to help with AML forensic analysis.",
                }

        # 3. Clearly off-topic (no AML keywords at all) — HARD BLOCK
        has_aml_context = any(kw in q_lower for kw in cls.AML_KEYWORDS)
        if not has_aml_context and len(q_lower.split()) > 3:
            return {
                "allowed": False,
                "reason":  "off_topic",
                "response": (
                    "🚫 Query blocked. FFRAG is restricted to financial forensics "
                    "and AML investigation only.\n\n"
                    "**In scope:** suspicious transactions, AML typologies, "
                    "SAR requirements, FATF/FinCEN regulations, account analysis.\n\n"
                    "**Your query does not match any of these topics.**"
                ),
            }

        # 4. Vague query warning (allowed but flagged)
        vague_patterns = [
            r"^(what|tell me|show me|explain)\??$",
            r"^(hello|hi|hey|help)\??$",
        ]
        for p in vague_patterns:
            if re.match(p, q_lower):
                return {
                    "allowed": True,
                    "warning": "Try being more specific — e.g. "
                               "'show structuring transactions' or "
                               "'explain the SAR filing timeline'.",
                }

        return {"allowed": True}

    @classmethod
    def check_output(cls, answer: str, contexts: list[str]) -> dict:
        """
        Validates the generated answer.
        Returns {"valid": True} or {"valid": False, "issue": "..."}
        """
        # 1. Answer too short — likely retrieval failure
        if len(answer.split()) < 20:
            return {
                "valid": False,
                "issue": "Answer too short — retrieval may have failed.",
            }

        # 2. Model refused to answer
        refusal_phrases = [
            "i cannot", "i can't", "i am not able", "i'm not able",
            "as an ai", "i don't have access",
        ]
        if any(p in answer.lower() for p in refusal_phrases):
            return {
                "valid":   False,
                "issue":   "Model declined to answer.",
            }

        # 3. System prompt leak check
        if "FFRAG" in answer and "system" in answer.lower():
            return {
                "valid": False,
                "issue": "Possible system prompt leak detected.",
            }

        return {"valid": True}


# ══════════════════════════════════════════════════════════════
# 4. RESPONSE FORMATTER + VARIATION
# ══════════════════════════════════════════════════════════════
class ResponseFormatter:
    """
    Formats the raw LLM answer into clean visual sections.
    Adds variation so repeated queries don't feel robotic.
    """

    # Section header variants — same meaning, different wording
    SECTION_VARIANTS = {
        "FINDINGS": [
            "📋 FINDINGS",
            "🔎 EVIDENCE",
            "📊 WHAT THE DATA SHOWS",
            "🗂 CASE FINDINGS",
        ],
        "TYPOLOGY MATCH": [
            "🏷 TYPOLOGY MATCH",
            "⚠️ PATTERN IDENTIFIED",
            "🔗 AML TYPOLOGY",
            "📌 CLASSIFICATION",
        ],
        "REGULATORY": [
            "📄 REGULATORY REFERENCE",
            "⚖️ APPLICABLE REGULATIONS",
            "📜 COMPLIANCE BASIS",
            "🏛 REGULATORY FRAMEWORK",
        ],
        "RISK VERDICT": [
            "🚨 RISK VERDICT",
            "⚡ ANALYST VERDICT",
            "🎯 RISK ASSESSMENT",
            "🔴 CASE VERDICT",
        ],
        "SOURCES USED": [
            "🔗 SOURCES",
            "📂 CITATIONS",
            "📎 EVIDENCE TRAIL",
            "🗃 REFERENCES",
        ],
    }

    # Intro phrase variants — prepended to answers for variation
    INTRO_VARIANTS = [
        "",  # no intro most of the time
        "",
        "",
        "**Forensic Analysis:**\n\n",
        "**Investigation Summary:**\n\n",
        "**AML Assessment:**\n\n",
    ]

    # Varied sentence starters injected before key sections
    PROSE_OPENERS = {
        "FINDINGS": [
            "The evidence shows that",
            "Analysis of the retrieved data indicates that",
            "Upon reviewing the records,",
            "The forensic data reveals that",
            "Investigation of the context confirms that",
        ],
        "RISK VERDICT": [
            "Based on the totality of evidence,",
            "Taking all factors into account,",
            "The combined signals suggest that",
            "From a forensic standpoint,",
            "Weighing the retrieved evidence,",
        ],
    }

    # HTML templates for each section — rendered via st.markdown unsafe_allow_html
    SECTION_HTML = {
        "FINDINGS":      '<div class="section-header findings">',
        "TYPOLOGY MATCH":'<div class="section-header typology">',
        "REGULATORY":    '<div class="section-header regulatory">',
        "RISK VERDICT":  '<div class="section-header verdict">',
        "SOURCES USED":  '<div class="section-header sources">',
    }

    SECTION_COLORS = {
        "FINDINGS":      "#4a9eff",   # blue
        "TYPOLOGY MATCH":"#f0c040",   # amber
        "REGULATORY":    "#c084f0",   # purple
        "RISK VERDICT":  "#ff8c42",   # orange
        "SOURCES USED":  "#3ddc84",   # green
    }

    @classmethod
    def format(cls, answer: str, seed: int = None) -> str:
        """
        Converts raw LLM answer into clean HTML-styled sections.
        Each section gets a coloured label bar + prose body.
        """
        if seed is not None:
            random.seed(seed)

        formatted = answer

        # ── Strip stray markdown intro lines like "**AML Assessment:**" ──
        formatted = re.sub(r"^\*\*[A-Za-z ]+:\*\*\s*\n?", "", formatted.strip())

        # ── Strip raw "chunk N" references ──
        formatted = re.sub(r"\(chunk\s*\d+\)", "", formatted)

        # ── Section patterns: numbered or unnumbered, any separator ──
        section_patterns = [
            (r"\d+\.\s*FINDINGS\s*[—:\-]+\s*",       "FINDINGS"),
            (r"\d+\.\s*TYPOLOGY MATCH\s*[—:\-]+\s*",  "TYPOLOGY MATCH"),
            (r"\d+\.\s*REGULATORY(?:\s+REFERENCE)?\s*[—:\-]+\s*", "REGULATORY"),
            (r"\d+\.\s*RISK VERDICT\s*[—:\-]+\s*",    "RISK VERDICT"),
            (r"\d+\.\s*SOURCES?\s*(?:USED)?\s*[—:\-]+\s*", "SOURCES USED"),
            (r"FINDINGS\s*[—:\-]+\s*",                "FINDINGS"),
            (r"TYPOLOGY MATCH\s*[—:\-]+\s*",          "TYPOLOGY MATCH"),
            (r"REGULATORY(?:\s+REFERENCE)?\s*[—:\-]+\s*", "REGULATORY"),
            (r"RISK VERDICT\s*[—:\-]+\s*",            "RISK VERDICT"),
            (r"SOURCES?\s*(?:USED)?\s*[—:\-]+\s*",   "SOURCES USED"),
            # Bold variants the LLM sometimes outputs
            (r"\*\*FINDINGS\*\*\s*\n",                "FINDINGS"),
            (r"\*\*TYPOLOGY MATCH\*\*\s*\n",          "TYPOLOGY MATCH"),
            (r"\*\*REGULATORY(?:\s+REFERENCE)?\*\*\s*\n", "REGULATORY"),
            (r"\*\*RISK VERDICT\*\*\s*\n",            "RISK VERDICT"),
            (r"\*\*(?:SOURCES?|CITATIONS?)\*\*\s*\n", "SOURCES USED"),
            # Emoji + bold variants seen in screenshots
            (r"[📋🔎📊🗂]\s*\*\*[^*]+\*\*\s*",        "FINDINGS"),
            (r"[🏷⚠️🔗📌]\s*\*\*[^*]+\*\*\s*",        "TYPOLOGY MATCH"),
            (r"[📄⚖️📜🏛]\s*\*\*[^*]+\*\*\s*",         "REGULATORY"),
            (r"[🚨⚡🎯🔴]\s*\*\*[^*]+\*\*\s*",          "RISK VERDICT"),
            (r"[🔗📂📎🗃]\s*\*\*[^*]+\*\*\s*",          "SOURCES USED"),
        ]

        for pattern, section_key in section_patterns:
            color    = cls.SECTION_COLORS.get(section_key, "#4a9eff")
            variants = cls.SECTION_VARIANTS.get(section_key, [section_key])
            label    = random.choice(variants).replace("📋","").replace("🔎","").replace(
                       "📊","").replace("🗂","").replace("🏷","").replace("⚠️","").replace(
                       "🔗","").replace("📌","").replace("📄","").replace("⚖️","").replace(
                       "📜","").replace("🏛","").replace("🚨","").replace("⚡","").replace(
                       "🎯","").replace("🔴","").replace("📂","").replace("📎","").replace(
                       "🗃","").strip()
            replacement = (
                f'\n\n<div style="margin:18px 0 6px;padding:6px 14px;'
                f'background:linear-gradient(90deg,{color}18,transparent);'
                f'border-left:3px solid {color};border-radius:0 6px 6px 0;">'
                f'<span style="font-family:IBM Plex Mono,monospace;font-size:10px;'
                f'font-weight:600;letter-spacing:2px;text-transform:uppercase;color:{color};">'
                f'{label}</span></div>\n'
            )
            formatted = re.sub(pattern, replacement, formatted, flags=re.IGNORECASE)

        # ── Prose opener variation for FINDINGS ──
        openers = cls.PROSE_OPENERS.get("FINDINGS", [])
        if openers:
            formatted = re.sub(
                r'(border-radius:0 6px 6px 0;">.*?</span></div>\n)([A-Z])',
                lambda m: m.group(1) + random.choice(openers) + " " + m.group(2).lower(),
                formatted, count=1
            )

        # ── Intro variation ──
        intro = random.choice(cls.INTRO_VARIANTS)
        if intro:
            clean_intro = re.sub(r"\*\*", "", intro).strip()
            if clean_intro:
                formatted = (
                    f'<div style="font-size:11px;color:#445577;font-family:'
                    f'IBM Plex Mono,monospace;letter-spacing:1px;margin-bottom:12px;">'
                    f'{clean_intro.upper()}</div>\n\n'
                ) + formatted

        # ── Clean up ──
        formatted = re.sub(r'\n{3,}', '\n\n', formatted).strip()

        return formatted

    @classmethod
    def format_guardrail_block(cls, reason: str, response: str) -> str:
        """Format a guardrail block message for display."""
        icons = {
            "off_topic":        "🔍",
            "injection_attempt": "⚠️",
            "empty_query":      "💬",
        }
        icon = icons.get(reason, "ℹ️")
        return f"{icon} {response}"

    @classmethod
    def format_warning(cls, warning: str) -> str:
        return f"💡 **Tip:** {warning}"
