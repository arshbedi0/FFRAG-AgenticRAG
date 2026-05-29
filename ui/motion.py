"""
motion.py — FFRAG Premium Motion Design System v2.0
Palantir x Bloomberg x Linear x Vercel AI x Perplexity
All animations run as pure CSS + Vanilla JS within Streamlit's injection model.
"""

# =============================================================================
# MOTION CSS — Complete style system (replaces app.py inline CSS block)
# =============================================================================

MOTION_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ══════════════════════════════════════════════════════════
   CSS CUSTOM PROPERTIES
   ══════════════════════════════════════════════════════════ */
:root {
  --ff-bg:          #040810;
  --ff-bg-2:        #060c18;
  --ff-bg-3:        #070d1c;
  --ff-blue:        #4a9eff;
  --ff-blue-dim:    #1a3a6a;
  --ff-teal:        #00d9e8;
  --ff-cyan:        #00f0ff;
  --ff-green:       #3ddc84;
  --ff-amber:       #f0c040;
  --ff-purple:      #c084f0;
  --ff-red:         #ff4444;
  --ff-orange:      #ff8c42;
  --ff-text:        #c8d0e0;
  --ff-text-mid:    #7a90a8;
  --ff-text-dim:    #334455;
  --ff-border:      #0f1e38;
  --ff-border-2:    #1a3050;

  --ease-out:    cubic-bezier(0.16, 1, 0.3, 1);
  --ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
  --ease-sharp:  cubic-bezier(0.4, 0, 0.2, 1);
  --dur-fast:    150ms;
  --dur-base:    250ms;
  --dur-slow:    450ms;
  --dur-enter:   600ms;
}

/* ══════════════════════════════════════════════════════════
   BASE RESET
   ══════════════════════════════════════════════════════════ */
html, body, [class*="css"] {
  font-family: 'IBM Plex Sans', sans-serif;
  background-color: var(--ff-bg);
  color: var(--ff-text);
}
.main { background-color: var(--ff-bg); }
.block-container { padding: 2rem 2rem 6rem; max-width: 1200px; }
hr { border-color: #0c1828 !important; }

/* ══════════════════════════════════════════════════════════
   PAGE LOAD — STAGED ENTRANCE SEQUENCE
   Stage 1: bg (0ms)  Stage 2: sidebar (120ms)
   Stage 3: header (200ms)  Stage 4: search (400ms)
   Stage 5: cards (300-600ms stagger)
   ══════════════════════════════════════════════════════════ */

@keyframes ff-bg-reveal {
  from { opacity: 0; }
  to   { opacity: 1; }
}
@keyframes ff-slide-right {
  from { opacity: 0; transform: translateX(-22px); }
  to   { opacity: 1; transform: translateX(0); }
}
@keyframes ff-slide-up {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes ff-slide-down {
  from { opacity: 0; transform: translateY(-12px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes ff-scale-in {
  from { opacity: 0; transform: scale(0.96); }
  to   { opacity: 1; transform: scale(1); }
}
@keyframes ff-fade-in {
  from { opacity: 0; }
  to   { opacity: 1; }
}

.main .block-container {
  animation: ff-bg-reveal 300ms var(--ease-out) both;
}
section[data-testid="stSidebar"] > div {
  animation: ff-slide-right 420ms 120ms var(--ease-out) both;
}
div[data-testid="stChatInput"] {
  animation: ff-slide-up 380ms 400ms var(--ease-out) both;
}

/* ══════════════════════════════════════════════════════════
   GRADIENT MESH BACKGROUND — Barely-there ambient motion
   ══════════════════════════════════════════════════════════ */

@keyframes ff-mesh-a {
  0%, 100% { transform: translate(0%, 0%) scale(1); }
  33%       { transform: translate(2.5%, 4%) scale(1.04); }
  66%       { transform: translate(-1.5%, 2.5%) scale(0.98); }
}
@keyframes ff-mesh-b {
  0%, 100% { transform: translate(0%, 0%) scale(1); }
  40%       { transform: translate(-3%, -2.5%) scale(1.03); }
  75%       { transform: translate(1.5%, 3.5%) scale(0.97); }
}
@keyframes ff-mesh-c {
  0%, 100% { transform: translate(0%, 0%) scale(1); }
  25%       { transform: translate(2.5%, -3%) scale(1.04); }
  65%       { transform: translate(-2%, 1.5%) scale(0.99); }
}

.ff-gradient-mesh {
  position: fixed;
  inset: 0;
  z-index: -3;
  pointer-events: none;
  overflow: hidden;
}
.ff-mesh-orb {
  position: absolute;
  border-radius: 50%;
  filter: blur(90px);
  opacity: 0.07;
  will-change: transform;
}
.ff-mesh-orb-1 {
  width: 660px; height: 660px;
  top: -220px; left: -120px;
  background: radial-gradient(circle, #153c7a 0%, transparent 70%);
  animation: ff-mesh-a 28s ease-in-out infinite;
}
.ff-mesh-orb-2 {
  width: 520px; height: 520px;
  top: 35%; right: -160px;
  background: radial-gradient(circle, #082840 0%, transparent 70%);
  animation: ff-mesh-b 34s ease-in-out infinite;
}
.ff-mesh-orb-3 {
  width: 440px; height: 440px;
  bottom: 8%; left: 28%;
  background: radial-gradient(circle, #0c2a55 0%, transparent 70%);
  animation: ff-mesh-c 22s ease-in-out infinite;
}

/* ══════════════════════════════════════════════════════════
   HEADER
   ══════════════════════════════════════════════════════════ */

.ffrag-logo {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 26px; font-weight: 600;
  color: var(--ff-blue); letter-spacing: -1px;
  animation: ff-slide-down 400ms 60ms var(--ease-out) both;
}
.ffrag-subtitle {
  font-size: 11px; color: var(--ff-text-dim);
  letter-spacing: 3px; text-transform: uppercase; margin-top: 2px;
  animation: ff-slide-down 400ms 110ms var(--ease-out) both;
}

/* ══════════════════════════════════════════════════════════
   SIDEBAR
   ══════════════════════════════════════════════════════════ */

section[data-testid="stSidebar"] {
  background: #030710 !important;
  border-right: 1px solid #0c1828;
}

/* Query suggestion buttons — hover lift + active left-border */
div[data-testid="stSidebar"] .stButton > button {
  position: relative;
  overflow: hidden;
  transition:
    transform var(--dur-fast) var(--ease-spring),
    background var(--dur-base) var(--ease-sharp),
    border-color var(--dur-base) var(--ease-sharp),
    box-shadow var(--dur-base) var(--ease-sharp) !important;
}
div[data-testid="stSidebar"] .stButton > button::before {
  content: '';
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 2px;
  background: var(--ff-blue);
  transform: scaleY(0);
  transform-origin: center;
  transition: transform 180ms var(--ease-out);
  border-radius: 0 1px 1px 0;
}
div[data-testid="stSidebar"] .stButton > button:hover::before {
  transform: scaleY(1);
}
div[data-testid="stSidebar"] .stButton > button:hover {
  transform: scale(1.02) !important;
  border-color: #243a5e !important;
  box-shadow: 0 4px 20px rgba(74, 158, 255, 0.07) !important;
}
div[data-testid="stSidebar"] .stButton > button:active {
  transform: scale(0.97) !important;
}

/* Sequential sidebar item entrance */
div[data-testid="stSidebar"] .stButton:nth-child(1)  { animation: ff-slide-up 320ms 210ms var(--ease-out) both; }
div[data-testid="stSidebar"] .stButton:nth-child(2)  { animation: ff-slide-up 320ms 260ms var(--ease-out) both; }
div[data-testid="stSidebar"] .stButton:nth-child(3)  { animation: ff-slide-up 320ms 310ms var(--ease-out) both; }
div[data-testid="stSidebar"] .stButton:nth-child(4)  { animation: ff-slide-up 320ms 360ms var(--ease-out) both; }
div[data-testid="stSidebar"] .stButton:nth-child(5)  { animation: ff-slide-up 320ms 410ms var(--ease-out) both; }
div[data-testid="stSidebar"] .stButton:nth-child(6)  { animation: ff-slide-up 320ms 460ms var(--ease-out) both; }
div[data-testid="stSidebar"] .stButton:nth-child(7)  { animation: ff-slide-up 320ms 510ms var(--ease-out) both; }
div[data-testid="stSidebar"] .stButton:nth-child(8)  { animation: ff-slide-up 320ms 560ms var(--ease-out) both; }
div[data-testid="stSidebar"] .stButton:nth-child(9)  { animation: ff-slide-up 320ms 610ms var(--ease-out) both; }
div[data-testid="stSidebar"] .stButton:nth-child(10) { animation: ff-slide-up 320ms 660ms var(--ease-out) both; }

/* ══════════════════════════════════════════════════════════
   METRIC CARDS — Data Source stats with count-up
   ══════════════════════════════════════════════════════════ */

.ff-metric-card {
  background: linear-gradient(135deg, #07101e 0%, #0a1628 100%);
  border: 1px solid var(--ff-border);
  border-radius: 10px;
  padding: 14px 16px;
  text-align: center;
  position: relative;
  overflow: hidden;
  transition:
    border-color var(--dur-base) var(--ease-sharp),
    transform var(--dur-fast) var(--ease-spring),
    box-shadow var(--dur-base) var(--ease-sharp);
}
.ff-metric-card::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(74,158,255,0.03) 0%, transparent 65%);
  opacity: 0;
  transition: opacity var(--dur-base) var(--ease-sharp);
  pointer-events: none;
}
.ff-metric-card:hover {
  border-color: var(--ff-border-2);
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(0,0,0,0.35), 0 0 14px rgba(74,158,255,0.04);
}
.ff-metric-card:hover::after { opacity: 1; }

.ff-metric-value {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 22px; font-weight: 600;
  color: var(--ff-blue);
  display: block; line-height: 1; margin-bottom: 4px;
}
.ff-metric-label {
  font-size: 10px; color: var(--ff-text-dim);
  text-transform: uppercase; letter-spacing: 1.5px;
}

/* Stagger metric card entrance */
.ff-metric-card:nth-child(1) { animation: ff-scale-in 380ms 280ms var(--ease-out) both; }
.ff-metric-card:nth-child(2) { animation: ff-scale-in 380ms 340ms var(--ease-out) both; }
.ff-metric-card:nth-child(3) { animation: ff-scale-in 380ms 400ms var(--ease-out) both; }
.ff-metric-card:nth-child(4) { animation: ff-scale-in 380ms 460ms var(--ease-out) both; }

/* Legacy metric-card (keep for compatibility) */
.metric-card {
  background: #07101e; border: 1px solid #0f1e38;
  border-radius: 8px; padding: 12px 16px; text-align: center;
  transition: border-color var(--dur-base) ease, transform var(--dur-fast) var(--ease-spring);
}
.metric-card:hover { border-color: var(--ff-border-2); transform: translateY(-1px); }
.metric-value {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 20px; font-weight: 600; color: var(--ff-blue);
}
.metric-label {
  font-size: 10px; color: var(--ff-text-dim);
  text-transform: uppercase; letter-spacing: 1.5px; margin-top: 2px;
}

/* ══════════════════════════════════════════════════════════
   SEARCH / CHAT INPUT — Premium focus experience
   ══════════════════════════════════════════════════════════ */

@keyframes ff-focus-ring {
  0%   { box-shadow: 0 0 0 0 rgba(74,158,255,0.18), 0 0 0 1px rgba(74,158,255,0.35); }
  50%  { box-shadow: 0 0 0 5px rgba(74,158,255,0), 0 0 0 1px rgba(74,158,255,0.5); }
  100% { box-shadow: 0 0 0 0 rgba(74,158,255,0), 0 0 0 1px rgba(74,158,255,0.35); }
}

@keyframes ff-input-idle {
  0%   { background-position: -200% center; }
  100% { background-position: 300% center; }
}

textarea[data-testid="stChatInputTextArea"] {
  background: #070d1c !important;
  color: var(--ff-text) !important;
  border: 1px solid #1a2e4a !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
  font-size: 14px !important;
  border-radius: 12px !important;
  transition:
    border-color var(--dur-base) var(--ease-sharp),
    box-shadow var(--dur-base) var(--ease-sharp) !important;
}
textarea[data-testid="stChatInputTextArea"]:hover {
  border-color: #243a5e !important;
  box-shadow: 0 0 0 1px rgba(74,158,255,0.08) !important;
}
textarea[data-testid="stChatInputTextArea"]:focus {
  border-color: var(--ff-blue) !important;
  box-shadow: 0 0 0 3px rgba(74,158,255,0.12), 0 0 24px rgba(74,158,255,0.05) !important;
  animation: ff-focus-ring 2.5s 0.4s ease-in-out infinite !important;
}

/* ══════════════════════════════════════════════════════════
   CHAT BUBBLES — Entrance + hover
   ══════════════════════════════════════════════════════════ */

@keyframes ff-bubble-in {
  0%   { opacity: 0; transform: translateY(20px) scale(0.97); filter: blur(6px); }
  55%  { opacity: 1; transform: translateY(-4px) scale(1.008); filter: blur(0); }
  100% { opacity: 1; transform: translateY(0) scale(1); }
}

@keyframes ff-scan-line {
  0%   { transform: translateY(-100%); opacity: 0.4; }
  60%  { opacity: 0.2; }
  100% { transform: translateY(400%); opacity: 0; }
}

.user-bubble {
  background: linear-gradient(135deg, #0d1d35 0%, #0a1628 100%);
  border: 1px solid #1e3a5f;
  border-radius: 14px 14px 2px 14px;
  padding: 14px 18px; margin: 12px 0 6px;
  font-size: 14px; color: #a8c0e0;
  animation: ff-bubble-in 480ms var(--ease-out) both !important;
  transition: border-color var(--dur-base) ease;
}
.user-bubble:hover { border-color: #2a4a7a; }

.assistant-bubble {
  background: linear-gradient(135deg, #060c18 0%, #07101e 100%);
  border: 1px solid #0f1e38;
  border-radius: 2px 14px 14px 14px;
  padding: 18px 22px; margin: 6px 0 12px;
  font-size: 14px; line-height: 1.8;
  position: relative; overflow: hidden;
  animation: ff-bubble-in 560ms var(--ease-out) both !important;
  transition:
    border-color var(--dur-base) ease,
    box-shadow var(--dur-base) ease;
}
.assistant-bubble::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent 0%, rgba(74,158,255,0.12) 50%, transparent 100%);
  animation: ff-scan-line 2.4s 0.3s ease-in-out 1;
}
.assistant-bubble:hover {
  border-color: #162840;
  box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}

/* ══════════════════════════════════════════════════════════
   TYPING CURSOR — for streaming / live text
   ══════════════════════════════════════════════════════════ */

@keyframes ff-cursor {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0; }
}
.ff-typing-cursor::after {
  content: '▋';
  font-size: 13px;
  color: var(--ff-blue);
  animation: ff-cursor 0.75s ease-in-out infinite;
  margin-left: 2px;
  vertical-align: baseline;
}

/* ══════════════════════════════════════════════════════════
   SOURCE BADGES
   ══════════════════════════════════════════════════════════ */

@keyframes ff-badge-pop {
  from { opacity: 0; transform: scale(0.82) translateY(5px); }
  to   { opacity: 1; transform: scale(1) translateY(0); }
}

.source-badges { display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0 0; }
.badge {
  font-family: 'IBM Plex Mono', monospace; font-size: 10px;
  padding: 3px 10px; border-radius: 20px; font-weight: 600;
  letter-spacing: 1px; text-transform: uppercase;
  transition: transform var(--dur-fast) var(--ease-spring), box-shadow var(--dur-fast) ease;
  animation: ff-badge-pop 280ms var(--ease-spring) both;
}
.source-badges .badge:nth-child(1) { animation-delay: 40ms; }
.source-badges .badge:nth-child(2) { animation-delay: 90ms; }
.source-badges .badge:nth-child(3) { animation-delay: 140ms; }
.badge:hover {
  transform: translateY(-1px) scale(1.05);
  box-shadow: 0 4px 14px rgba(0,0,0,0.35);
}
.badge-txn   { background: #0a1f15; color: #3ddc84; border: 1px solid #1a4a30; }
.badge-graph { background: #1a170a; color: #f0c040; border: 1px solid #3a3000; }
.badge-reg   { background: #180d1f; color: #c084f0; border: 1px solid #3a1555; }

/* ══════════════════════════════════════════════════════════
   SCORE BLOCK
   ══════════════════════════════════════════════════════════ */

@keyframes ff-score-enter {
  from { opacity: 0; transform: translateX(-10px); }
  to   { opacity: 1; transform: translateX(0); }
}

.score-block {
  display: inline-flex; align-items: center; gap: 10px;
  background: #060c18; border: 1px solid #1e3050;
  border-radius: 8px; padding: 8px 16px; margin-top: 12px;
  font-family: 'IBM Plex Mono', monospace; font-size: 12px;
  animation: ff-score-enter 380ms 180ms var(--ease-out) both !important;
  transition: box-shadow var(--dur-base) ease;
}
.score-block:hover { box-shadow: 0 0 14px rgba(74,158,255,0.08); }
.score-critical { color: #ff4444; border-color: #3a0000; }
.score-high     { color: #ff8c42; border-color: #3a1500; }
.score-medium   { color: #f0c040; border-color: #3a2800; }
.score-low      { color: #3ddc84; border-color: #003a18; }

/* ══════════════════════════════════════════════════════════
   SYSTEM BLOCKS
   ══════════════════════════════════════════════════════════ */

@keyframes ff-shake {
  0%, 100% { transform: translateX(0); }
  20%       { transform: translateX(-4px); }
  40%       { transform: translateX(4px); }
  60%       { transform: translateX(-2px); }
  80%       { transform: translateX(2px); }
}

.guardrail-block {
  background: #160c0c; border: 1px solid #3a1515;
  border-left: 3px solid #ff4444;
  border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 8px 0;
  font-size: 13px; color: #cc8888;
  animation: ff-slide-up 280ms var(--ease-out) both,
             ff-shake 400ms 300ms ease both !important;
}
.warning-block {
  background: #0a1510; border: 1px solid #163020;
  border-left: 3px solid #3ddc84;
  border-radius: 0 8px 8px 0; padding: 10px 14px; margin: 8px 0;
  font-size: 12px; color: #5aaa70;
  animation: ff-slide-up 280ms var(--ease-out) both !important;
}
.tip-box {
  background: #0a1510; border: 1px solid #163020;
  border-left: 3px solid #3ddc84;
  border-radius: 0 8px 8px 0; padding: 10px 14px; margin: 8px 0;
  font-size: 12px; color: #5aaa70;
}

/* ══════════════════════════════════════════════════════════
   AGENT BOX
   ══════════════════════════════════════════════════════════ */

@keyframes ff-agent-scan {
  0%   { background-position: -150% 0; }
  100% { background-position: 250% 0; }
}

.agent-box {
  background: #070d1c; border: 1px solid #152840;
  border-left: 3px solid var(--ff-blue);
  border-radius: 0 8px 8px 0; padding: 10px 14px; margin: 4px 0 10px;
  font-size: 11px; color: #3a6080;
  font-family: 'IBM Plex Mono', monospace;
  position: relative; overflow: hidden;
  animation: ff-slide-up 280ms var(--ease-out) both !important;
}
.agent-box::after {
  content: '';
  position: absolute; top: 0; left: 0; width: 100%; height: 100%;
  background: linear-gradient(90deg,
    transparent 0%,
    rgba(74,158,255,0.035) 50%,
    transparent 100%);
  background-size: 200% 100%;
  animation: ff-agent-scan 4s 0.5s ease-in-out infinite;
  pointer-events: none;
}

/* ══════════════════════════════════════════════════════════
   VOICE PANEL
   ══════════════════════════════════════════════════════════ */

.voice-panel {
  background: linear-gradient(135deg, #0c0614 0%, #080414 100%);
  border: 1px solid #2a1545; border-radius: 16px;
  padding: 0; margin: 16px 0 8px; overflow: hidden; position: relative;
}
.voice-panel-header {
  background: linear-gradient(90deg, #1a0a2e 0%, #0c0614 100%);
  border-bottom: 1px solid #2a1545;
  padding: 10px 18px; display: flex; align-items: center; gap: 10px;
}
.voice-panel-title {
  font-family: 'IBM Plex Mono', monospace; font-size: 10px; font-weight: 600;
  color: #9060cc; letter-spacing: 3px; text-transform: uppercase;
}
.voice-panel-body { padding: 16px 18px 14px; }
.voice-dot { width: 8px; height: 8px; border-radius: 50%; background: #9060cc; box-shadow: 0 0 8px #9060cc88; display: inline-block; }
.voice-dot-live { background: #ff4060; box-shadow: 0 0 10px #ff406088; animation: ff-pulse-dot 1s ease-in-out infinite; }
@keyframes ff-pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.5; transform: scale(0.85); }
}
.voice-transcript-box {
  background: #0a0518; border: 1px solid #2a1545; border-radius: 10px;
  padding: 12px 16px; margin-top: 10px;
  font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #c0a0f0;
  line-height: 1.6; min-height: 44px; position: relative;
}
.voice-transcript-label { font-size: 9px; color: #5a3a88; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 4px; }
.voice-send-hint { font-size: 10px; color: #5a3a88; font-family: 'IBM Plex Mono', monospace; margin-top: 8px; text-align: center; letter-spacing: 1px; }
.voice-error { background: #120808; border: 1px solid #3a1515; border-radius: 8px; padding: 10px 14px; margin-top: 8px; font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #cc5555; }

/* ══════════════════════════════════════════════════════════
   BUTTONS — All contexts
   ══════════════════════════════════════════════════════════ */

.stButton > button {
  background: #0a1428 !important;
  border: 1px solid #1a3050 !important;
  color: #7aacdd !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 11px !important; letter-spacing: 1px !important;
  border-radius: 6px !important;
  transition:
    transform var(--dur-fast) var(--ease-spring),
    background var(--dur-base) var(--ease-sharp),
    border-color var(--dur-base) var(--ease-sharp),
    box-shadow var(--dur-base) var(--ease-sharp),
    color var(--dur-base) var(--ease-sharp) !important;
}
.stButton > button:hover {
  background: #0f1e38 !important;
  border-color: #4a9eff !important;
  color: #4a9eff !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 16px rgba(74,158,255,0.1) !important;
}
.stButton > button:active {
  transform: scale(0.96) !important;
  box-shadow: none !important;
}

/* ══════════════════════════════════════════════════════════
   RESPONSE SECTION HEADERS — Animated reveals
   ══════════════════════════════════════════════════════════ */

@keyframes ff-section-in {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes ff-border-grow {
  from { width: 0%; }
  to   { width: 100%; }
}
@keyframes ff-glow-sweep {
  0%   { background-position: -200% center; }
  100% { background-position: 300% center; }
}

.ff-section {
  position: relative;
  margin: 18px 0 8px;
  padding: 6px 14px;
  border-radius: 0 6px 6px 0;
  overflow: hidden;
  animation: ff-section-in 380ms var(--ease-out) both;
}
.ff-section::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0;
  height: 1px; width: 0%;
  animation: ff-border-grow 550ms 200ms var(--ease-out) both;
}
.ff-section-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px; font-weight: 600;
  letter-spacing: 2.5px; text-transform: uppercase;
  position: relative; z-index: 1;
}
.ff-section-label::before {
  content: '';
  position: absolute; top: -4px; left: -4px; right: -4px; bottom: -4px;
  background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.06) 50%, transparent 100%);
  background-size: 300% 100%;
  animation: ff-glow-sweep 1s 0.25s ease-out both;
  pointer-events: none;
}

.ff-section-findings  { background: linear-gradient(90deg, rgba(74,158,255,0.07), transparent); border-left: 3px solid #4a9eff; }
.ff-section-findings .ff-section-label { color: #4a9eff; }
.ff-section-findings::after { background: rgba(74,158,255,0.25); }

.ff-section-typology  { background: linear-gradient(90deg, rgba(240,192,64,0.07), transparent); border-left: 3px solid #f0c040; }
.ff-section-typology .ff-section-label { color: #f0c040; }
.ff-section-typology::after { background: rgba(240,192,64,0.25); }

.ff-section-regulatory { background: linear-gradient(90deg, rgba(192,132,240,0.07), transparent); border-left: 3px solid #c084f0; }
.ff-section-regulatory .ff-section-label { color: #c084f0; }
.ff-section-regulatory::after { background: rgba(192,132,240,0.25); }

.ff-section-verdict { background: linear-gradient(90deg, rgba(255,140,66,0.07), transparent); border-left: 3px solid #ff8c42; }
.ff-section-verdict .ff-section-label { color: #ff8c42; }
.ff-section-verdict::after { background: rgba(255,140,66,0.25); }

.ff-section-sources { background: linear-gradient(90deg, rgba(61,220,132,0.07), transparent); border-left: 3px solid #3ddc84; }
.ff-section-sources .ff-section-label { color: #3ddc84; }
.ff-section-sources::after { background: rgba(61,220,132,0.25); }

/* ══════════════════════════════════════════════════════════
   AI THINKING — Graph intelligence loader
   ══════════════════════════════════════════════════════════ */

@keyframes ff-node-orbit-1 {
  from { transform: rotate(0deg)   translateX(28px) rotate(0deg); }
  to   { transform: rotate(360deg) translateX(28px) rotate(-360deg); }
}
@keyframes ff-node-orbit-2 {
  from { transform: rotate(120deg)  translateX(20px) rotate(-120deg); }
  to   { transform: rotate(480deg)  translateX(20px) rotate(-480deg); }
}
@keyframes ff-node-orbit-3 {
  from { transform: rotate(240deg)  translateX(35px) rotate(-240deg); }
  to   { transform: rotate(600deg)  translateX(35px) rotate(-600deg); }
}
@keyframes ff-ring-breathe {
  0%, 100% { transform: translate(-50%,-50%) scale(0.92); opacity: 0.45; }
  50%       { transform: translate(-50%,-50%) scale(1.08); opacity: 0.18; }
}
@keyframes ff-core-glow {
  0%, 100% { box-shadow: 0 0 0 0 rgba(74,158,255,0.45), 0 0 8px rgba(74,158,255,0.3); }
  50%       { box-shadow: 0 0 0 6px rgba(74,158,255,0), 0 0 18px rgba(74,158,255,0.55); }
}
@keyframes ff-ellipsis {
  0%   { content: "."; }
  33%  { content: ".."; }
  66%  { content: "..."; }
  100% { content: "."; }
}

.ff-thinking {
  display: flex; align-items: center; gap: 18px;
  padding: 16px 20px;
  background: #060c18; border: 1px solid #0f1e38;
  border-left: 3px solid var(--ff-blue);
  border-radius: 0 10px 10px 0; margin: 12px 0;
  animation: ff-slide-up 280ms var(--ease-out) both;
}
.ff-thinking-graph {
  position: relative; width: 64px; height: 64px; flex-shrink: 0;
}
.ff-thinking-core {
  position: absolute; top: 50%; left: 50%;
  width: 10px; height: 10px; border-radius: 50%;
  background: var(--ff-blue);
  transform: translate(-50%,-50%);
  animation: ff-core-glow 1.6s ease-in-out infinite;
}
.ff-thinking-ring {
  position: absolute; border: 1px solid rgba(74,158,255,0.18);
  border-radius: 50%;
}
.ff-thinking-ring-1 { width: 26px; height: 26px; top: 50%; left: 50%; animation: ff-ring-breathe 2s 0s    ease-in-out infinite; }
.ff-thinking-ring-2 { width: 42px; height: 42px; top: 50%; left: 50%; animation: ff-ring-breathe 2s 0.4s  ease-in-out infinite; }
.ff-thinking-ring-3 { width: 58px; height: 58px; top: 50%; left: 50%; animation: ff-ring-breathe 2s 0.8s  ease-in-out infinite; }
.ff-thinking-node {
  position: absolute; top: 50%; left: 50%;
  border-radius: 50%; margin: -2.5px;
}
.ff-thinking-node-1 { width: 5px; height: 5px; background: var(--ff-blue);   animation: ff-node-orbit-1 2s   linear infinite; }
.ff-thinking-node-2 { width: 5px; height: 5px; background: var(--ff-teal);   animation: ff-node-orbit-2 2.7s linear infinite; }
.ff-thinking-node-3 { width: 4px; height: 4px; background: var(--ff-green);  animation: ff-node-orbit-3 3.3s linear infinite; }

.ff-thinking-text { font-family: 'IBM Plex Mono', monospace; }
.ff-thinking-header {
  font-size: 10px; color: var(--ff-blue);
  letter-spacing: 3px; text-transform: uppercase;
  display: flex; align-items: center; gap: 6px; margin-bottom: 5px;
}
.ff-thinking-header::before {
  content: '';
  display: inline-block; width: 5px; height: 5px; border-radius: 50%;
  background: var(--ff-blue); box-shadow: 0 0 7px rgba(74,158,255,0.7);
  animation: ff-pulse-dot 1s ease-in-out infinite;
}
.ff-thinking-stage {
  font-size: 12px; color: var(--ff-text-mid);
  transition: color 300ms ease;
}

/* ══════════════════════════════════════════════════════════
   RETRIEVAL TIMELINE
   ══════════════════════════════════════════════════════════ */

@keyframes ff-dot-ping {
  0%, 100% { transform: scale(1); opacity: 1; }
  50%       { transform: scale(1.45); opacity: 0.65; }
}
@keyframes ff-line-flow {
  0%   { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
@keyframes ff-stage-appear {
  from { opacity: 0; transform: translateX(-8px); }
  to   { opacity: 1; transform: translateX(0); }
}

.ff-retrieval {
  background: #060c18; border: 1px solid #0f1e38;
  border-left: 3px solid var(--ff-blue);
  border-radius: 0 10px 10px 0; padding: 16px 20px; margin: 12px 0;
  animation: ff-slide-up 300ms var(--ease-out) both;
}
.ff-retrieval-header {
  font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--ff-blue);
  letter-spacing: 3px; text-transform: uppercase;
  display: flex; align-items: center; gap: 8px; margin-bottom: 14px;
}
.ff-retrieval-header::before {
  content: ''; display: inline-block; width: 6px; height: 6px; border-radius: 50%;
  background: var(--ff-blue); box-shadow: 0 0 8px rgba(74,158,255,0.6);
  animation: ff-dot-ping 1s ease-in-out infinite;
}

.ff-stage {
  display: flex; align-items: center; gap: 12px;
  padding: 5px 0; font-size: 12px; color: var(--ff-text-dim);
  opacity: 0.3; transition: opacity 300ms ease, color 300ms ease;
}
.ff-stage.ff-active { opacity: 1; color: var(--ff-text); }
.ff-stage.ff-done   { opacity: 0.65; color: var(--ff-blue); }

.ff-stage-dot {
  width: 8px; height: 8px; border-radius: 50%;
  border: 1px solid #1a3050; flex-shrink: 0;
  transition: all 300ms var(--ease-spring);
}
.ff-stage.ff-active .ff-stage-dot {
  background: var(--ff-blue); border-color: var(--ff-blue);
  box-shadow: 0 0 8px rgba(74,158,255,0.55);
  animation: ff-dot-ping 0.85s ease-in-out infinite;
}
.ff-stage.ff-done .ff-stage-dot {
  background: var(--ff-green); border-color: var(--ff-green);
  box-shadow: 0 0 6px rgba(61,220,132,0.35);
}
.ff-stage-track {
  flex: 1; height: 1px; background: #0f1e38; border-radius: 1px; overflow: hidden;
}
.ff-stage.ff-active .ff-stage-track::after {
  content: ''; display: block; height: 100%;
  background: linear-gradient(90deg, transparent, var(--ff-blue), transparent);
  background-size: 200% 100%;
  animation: ff-line-flow 1.4s linear infinite;
}
.ff-stage-code { font-family: 'IBM Plex Mono', monospace; font-size: 9px; color: #1e3050; letter-spacing: 1px; }

/* ══════════════════════════════════════════════════════════
   SCROLL REVEAL
   ══════════════════════════════════════════════════════════ */

.ff-scroll-hidden  { opacity: 0; transform: translateY(20px); transition: opacity 300ms var(--ease-out), transform 300ms var(--ease-out); }
.ff-scroll-visible { opacity: 1; transform: translateY(0); }

/* ══════════════════════════════════════════════════════════
   TABS
   ══════════════════════════════════════════════════════════ */

button[data-baseweb="tab"] {
  transition: color 150ms ease, border-color 150ms ease !important;
}
div[data-testid="stTabs"] { animation: ff-fade-in 300ms 200ms var(--ease-out) both; }

/* ══════════════════════════════════════════════════════════
   EXPANDERS
   ══════════════════════════════════════════════════════════ */

details[data-testid="stExpander"] {
  transition: all 200ms var(--ease-out);
  border-color: #0f1e38 !important;
}

/* ══════════════════════════════════════════════════════════
   FIRE ANIMATION — Query bar ignition on submit
   ══════════════════════════════════════════════════════════ */

/* Border flickers through fire palette then cools back */
@keyframes ff-fire-border {
  0%   {
    border-color: #ff9900 !important;
    box-shadow: 0 0 0 2px rgba(255,153,0,0.3),
                0 0 14px rgba(255,100,0,0.35),
                inset 0 0 8px rgba(255,120,0,0.08) !important;
  }
  18%  {
    border-color: #ff4400 !important;
    box-shadow: 0 0 0 4px rgba(255,68,0,0.22),
                0 0 32px rgba(255,50,0,0.5),
                inset 0 0 14px rgba(255,80,0,0.1) !important;
  }
  42%  {
    border-color: #ffcc00 !important;
    box-shadow: 0 0 0 5px rgba(255,200,0,0.18),
                0 0 44px rgba(255,160,0,0.42),
                inset 0 0 12px rgba(255,200,0,0.08) !important;
  }
  68%  {
    border-color: #ff6600 !important;
    box-shadow: 0 0 0 3px rgba(255,102,0,0.14),
                0 0 22px rgba(255,80,0,0.28) !important;
  }
  100% {
    border-color: #1a2e4a !important;
    box-shadow: none !important;
  }
}

/* Spark: rises and fades, with horizontal drift via --dx */
@keyframes ff-spark-rise {
  0%   { transform: translateY(0)    translateX(0)           scale(1);    opacity: 0.95; }
  55%  { transform: translateY(-50px) translateX(var(--dx,0px)) scale(0.55); opacity: 0.65; }
  100% { transform: translateY(-90px) translateX(var(--dx,0px)) scale(0);   opacity: 0;    }
}

/* Flicker on the spark body itself */
@keyframes ff-spark-flicker {
  0%, 100% { opacity: 1;    filter: brightness(1); }
  30%       { opacity: 0.7;  filter: brightness(1.4); }
  60%       { opacity: 0.9;  filter: brightness(0.85); }
}

/* Underline ember — a brief horizontal glow line under the input */
@keyframes ff-ember-line {
  0%   { transform: scaleX(0);   opacity: 0; }
  15%  { transform: scaleX(1);   opacity: 1; }
  70%  { transform: scaleX(1);   opacity: 0.6; }
  100% { transform: scaleX(1.05); opacity: 0; }
}

/* One-shot ignition (kept for compatibility) */
div[data-testid="stChatInput"].ff-on-fire
  textarea[data-testid="stChatInputTextArea"] {
  animation: ff-fire-border 1.15s ease-out forwards !important;
}

/* ── Infinite continuous fire glow ── */
@keyframes ff-fire-border-loop {
  0%, 100% {
    border-color: #1e3a5f !important;
    box-shadow: 0 0 6px rgba(255,80,0,0.1) !important;
  }
  22% {
    border-color: #ff6600 !important;
    box-shadow: 0 0 0 2px rgba(255,102,0,0.2),
                0 0 22px rgba(255,80,0,0.32) !important;
  }
  50% {
    border-color: #ffaa00 !important;
    box-shadow: 0 0 0 3px rgba(255,170,0,0.16),
                0 0 34px rgba(255,140,0,0.36) !important;
  }
  76% {
    border-color: #ff3300 !important;
    box-shadow: 0 0 0 2px rgba(255,51,0,0.18),
                0 0 20px rgba(255,50,0,0.28) !important;
  }
}

div[data-testid="stChatInput"].ff-fire-infinite
  textarea[data-testid="stChatInputTextArea"] {
  animation: ff-fire-border-loop 2.8s ease-in-out infinite !important;
}

/* Ember underline pseudo-element via a sibling div we inject */
.ff-ember-line {
  position: absolute;
  bottom: 0; left: 5%; right: 5%;
  height: 2px;
  border-radius: 2px;
  background: linear-gradient(90deg,
    transparent 0%,
    #ff4400 20%,
    #ffcc00 50%,
    #ff4400 80%,
    transparent 100%);
  transform-origin: center;
  animation: ff-ember-line 900ms ease-out forwards;
  pointer-events: none;
  z-index: 9998;
}

/* Individual spark particle */
.ff-spark {
  position: absolute;
  border-radius: 50% 50% 38% 38%;
  pointer-events: none;
  z-index: 9999;
  mix-blend-mode: screen;
  animation:
    ff-spark-rise    var(--dur, 680ms) ease-out        forwards,
    ff-spark-flicker calc(var(--dur, 680ms) * 0.45) ease-in-out infinite;
}

@media (prefers-reduced-motion: reduce) {
  .ff-spark, .ff-ember-line { display: none !important; }
}

/* ══════════════════════════════════════════════════════════
   REDUCE MOTION — Accessibility
   ══════════════════════════════════════════════════════════ */

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration:       0.01ms !important;
    animation-iteration-count: 1    !important;
    transition-duration:       0.01ms !important;
  }
}
</style>"""


# =============================================================================
# BACKGROUND CANVAS — Particle system + Graph network (injected once)
# =============================================================================

BACKGROUND_CANVAS = """
<div id="ff-bg-layer" style="
  position:fixed; inset:0; z-index:-2;
  pointer-events:none; overflow:hidden;
">
  <canvas id="ff-canvas" style="
    position:absolute; inset:0;
    width:100%; height:100%;
  "></canvas>
</div>
<div class="ff-gradient-mesh">
  <div class="ff-mesh-orb ff-mesh-orb-1"></div>
  <div class="ff-mesh-orb ff-mesh-orb-2"></div>
  <div class="ff-mesh-orb ff-mesh-orb-3"></div>
</div>

<script>
(function () {
  'use strict';

  // Cancel any previous animation loop (Streamlit re-renders)
  if (window._ffRaf) { cancelAnimationFrame(window._ffRaf); window._ffRaf = null; }

  const canvas = document.getElementById('ff-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  let W, H, particles, nodes, edges;
  let lastTs = 0;
  const FRAME_MS = 1000 / 28;   // ~28 FPS cap — smooth but light

  /* ── Resize ── */
  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  /* ── Particle factory ── */
  function mkParticle() {
    const h = Math.random() < 0.5;
    let x, y, vx, vy;
    if (h) {
      x  = Math.random() < 0.5 ? 0 : W;
      y  = Math.random() * H;
      vx = x === 0 ? (Math.random() * 0.25 + 0.05) : -(Math.random() * 0.25 + 0.05);
      vy = (Math.random() - 0.5) * 0.15;
    } else {
      x  = Math.random() * W;
      y  = Math.random() < 0.5 ? 0 : H;
      vy = y === 0 ? (Math.random() * 0.25 + 0.05) : -(Math.random() * 0.25 + 0.05);
      vx = (Math.random() - 0.5) * 0.15;
    }
    const COLS = ['#4a9eff','#00d9e8','#3ddc84','#f0c040','#c084f0'];
    return {
      x, y, vx, vy,
      r:       Math.random() * 1.1 + 0.4,
      alpha:   Math.random() * 0.28 + 0.06,
      color:   COLS[Math.floor(Math.random() * COLS.length)],
      life: 0, maxLife: Math.random() * 700 + 200,
    };
  }

  /* ── Node factory ── */
  function mkNode() {
    return {
      x: Math.random() * W, y: Math.random() * H,
      r:  Math.random() * 1.8 + 0.8,
      vx: (Math.random() - 0.5) * 0.06,
      vy: (Math.random() - 0.5) * 0.06,
      alpha: Math.random() * 0.13 + 0.03,
      ph: Math.random() * Math.PI * 2,
      ps: Math.random() * 0.012 + 0.004,
    };
  }

  /* ── Init ── */
  function init() {
    resize();
    particles = Array.from({ length: 20 }, mkParticle);
    nodes     = Array.from({ length: 18 }, mkNode);
    edges     = [];
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        if (Math.random() < 0.22) {
          edges.push({ a: i, b: j,
            ph: Math.random() * Math.PI * 2,
            ps: Math.random() * 0.007 + 0.002 });
        }
      }
    }
  }

  /* ── Draw graph network ── */
  function drawGraph(dt) {
    for (const n of nodes) {
      n.x  += n.vx * dt * 0.045; n.y += n.vy * dt * 0.045;
      n.ph += n.ps * dt * 0.1;
      if (n.x < 0 || n.x > W) { n.vx *= -1; n.x = Math.max(0, Math.min(W, n.x)); }
      if (n.y < 0 || n.y > H) { n.vy *= -1; n.y = Math.max(0, Math.min(H, n.y)); }
    }
    for (const e of edges) {
      e.ph += e.ps * dt * 0.1;
      const na = nodes[e.a], nb = nodes[e.b];
      const d  = Math.hypot(nb.x - na.x, nb.y - na.y);
      if (d > 280) continue;
      const ea = (Math.sin(e.ph) * 0.5 + 0.5) * 0.055 + 0.008;
      ctx.beginPath();
      ctx.moveTo(na.x, na.y); ctx.lineTo(nb.x, nb.y);
      ctx.strokeStyle = `rgba(74,158,255,${ea.toFixed(3)})`;
      ctx.lineWidth   = 0.5; ctx.stroke();
    }
    for (const n of nodes) {
      const pa = (Math.sin(n.ph) * 0.5 + 0.5) * n.alpha + 0.018;
      ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(74,158,255,${pa.toFixed(3)})`; ctx.fill();
    }
  }

  /* ── Draw particles ── */
  function drawParticles(dt) {
    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i];
      p.x    += p.vx * dt * 0.055; p.y += p.vy * dt * 0.055; p.life += dt;
      const t = p.life / p.maxLife;
      const a = t < 0.1 ? (t / 0.1) * p.alpha
              : t > 0.82 ? ((1 - t) / 0.18) * p.alpha : p.alpha;
      const hex = Math.floor(a * 255).toString(16).padStart(2, '0');
      ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = p.color + hex; ctx.fill();
      if (p.life >= p.maxLife || p.x < -12 || p.x > W + 12 || p.y < -12 || p.y > H + 12) {
        particles[i] = mkParticle();
      }
    }
  }

  /* ── Render loop ── */
  function frame(ts) {
    const dt = ts - lastTs;
    if (dt < FRAME_MS) { window._ffRaf = requestAnimationFrame(frame); return; }
    lastTs = ts;
    ctx.clearRect(0, 0, W, H);
    drawGraph(dt); drawParticles(dt);
    window._ffRaf = requestAnimationFrame(frame);
  }

  /* ── Scroll observer (re-init each render) ── */
  function setupScrollObserver() {
    if (!window.IntersectionObserver) return;
    const obs = new IntersectionObserver(
      entries => entries.forEach(e => {
        if (e.isIntersecting) {
          e.target.classList.replace('ff-scroll-hidden', 'ff-scroll-visible');
        }
      }),
      { threshold: 0.12, rootMargin: '0px 0px -32px 0px' }
    );
    document.querySelectorAll('.ff-scroll-reveal').forEach(el => {
      el.classList.add('ff-scroll-hidden');
      obs.observe(el);
    });
  }

  /* ── Count-up animation (triggered after short delay) ── */
  function countUp(el, target, duration) {
    const start = performance.now();
    (function tick(now) {
      const p = Math.min((now - start) / duration, 1);
      const e = 1 - Math.pow(1 - p, 3);
      el.textContent = Math.round(target * e).toLocaleString();
      if (p < 1) requestAnimationFrame(tick);
    })(performance.now());
  }
  window._ffCountUp = function () {
    document.querySelectorAll('[data-ff-count]').forEach(el => {
      const t = parseFloat(el.dataset.ffCount.replace(/,/g, '')) || 0;
      countUp(el, t, 1100);
    });
  };

  /* ── Spotlight effect ── */
  document.addEventListener('mousemove', e => {
    document.querySelectorAll('.ff-spotlight').forEach(card => {
      const r = card.getBoundingClientRect();
      card.style.setProperty('--sx', (e.clientX - r.left) + 'px');
      card.style.setProperty('--sy', (e.clientY - r.top)  + 'px');
    });
  });

  /* ── Resize + visibility ── */
  window.addEventListener('resize', resize);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      cancelAnimationFrame(window._ffRaf);
    } else {
      lastTs = 0; window._ffRaf = requestAnimationFrame(frame);
    }
  });

  /* ── Bootstrap ── */
  function start() {
    init();
    window._ffRaf = requestAnimationFrame(frame);
    setTimeout(() => {
      setupScrollObserver();
      window._ffCountUp();
    }, 550);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
</script>
"""


# =============================================================================
# COMPONENT HELPERS
# =============================================================================

def animated_metric_card(label: str, value: str, nth: int = 0) -> str:
    """Metric card with count-up animation. value should be numeric string like '1,000'."""
    raw = value.replace(",", "").replace(".", "")
    delay = nth * 75
    return (
        f'<div class="ff-metric-card" style="animation-delay:{delay}ms;">'
        f'  <span class="ff-metric-value" data-ff-count="{raw}">{value}</span>'
        f'  <span class="ff-metric-label">{label}</span>'
        f'</div>'
    )


def get_ai_thinking_html(stage_text: str = "Routing query...") -> str:
    """Graph-intelligence loader — replaces Streamlit spinner."""
    return (
        '<div class="ff-thinking">'
        '  <div class="ff-thinking-graph">'
        '    <div class="ff-thinking-ring ff-thinking-ring-1"></div>'
        '    <div class="ff-thinking-ring ff-thinking-ring-2"></div>'
        '    <div class="ff-thinking-ring ff-thinking-ring-3"></div>'
        '    <div class="ff-thinking-core"></div>'
        '    <div class="ff-thinking-node ff-thinking-node-1"></div>'
        '    <div class="ff-thinking-node ff-thinking-node-2"></div>'
        '    <div class="ff-thinking-node ff-thinking-node-3"></div>'
        '  </div>'
        '  <div class="ff-thinking-text">'
        '    <div class="ff-thinking-header">FFRAG &nbsp;·&nbsp; AGENTIC MODE</div>'
        f'   <div class="ff-thinking-stage">{stage_text}</div>'
        '  </div>'
        '</div>'
    )


_RETRIEVAL_STAGES = [
    ("Routing query",           "ROUTER"),
    ("Expanding context",       "QUERY EXP"),
    ("Retrieving transactions", "TXN STORE"),
    ("Graph analysis",          "GRAPH DB"),
    ("Generating response",     "LLM"),
]

def get_retrieval_timeline(active: int = 0) -> str:
    """Animated retrieval pipeline timeline. active=0..4 sets which stage is live."""
    rows = ""
    for i, (label, code) in enumerate(_RETRIEVAL_STAGES):
        if i < active:
            cls = "ff-stage ff-done"
        elif i == active:
            cls = "ff-stage ff-active"
        else:
            cls = "ff-stage"
        rows += (
            f'<div class="{cls}" style="animation-delay:{i*55}ms;">'
            f'  <div class="ff-stage-dot"></div>'
            f'  <span>{label}</span>'
            f'  <div class="ff-stage-track"></div>'
            f'  <span class="ff-stage-code">{code}</span>'
            f'</div>'
        )
    return (
        '<div class="ff-retrieval">'
        '  <div class="ff-retrieval-header">AGENTIC PIPELINE</div>'
        + rows +
        '</div>'
    )


def get_fire_trigger_js() -> str:
    """
    Returns a full HTML page for use with st.components.v1.html(height=0).
    The iframe reaches into the parent Streamlit document via window.parent
    and fires the animation on the chat input bar.
    st.markdown <script> tags are silently dropped by React on reruns —
    st.components.v1.html() is the only reliable way to execute JS each rerun.
    """
    return """<!DOCTYPE html><html><body style="margin:0;padding:0;overflow:hidden;">
<script>
(function () {
  var COLORS = [
    ['#ff3300','#ff7700'],
    ['#ff5500','#ffaa00'],
    ['#ff1100','#ff5500'],
    ['#ffbb00','#ff8800'],
    ['#ff7700','#ffdd00'],
    ['#ff4400','#ff9900'],
  ];

  function spawnSpark(wrap, doc) {
    var c     = COLORS[Math.floor(Math.random() * COLORS.length)];
    var size  = 2 + Math.random() * 5;
    var xPct  = 3 + Math.random() * 94;
    var dx    = (Math.random() - 0.5) * 42;
    var dur   = 900 + Math.random() * 800;   // slower: 900–1700 ms
    var delay = 80 + Math.random() * 380;    // staggered start

    var s = doc.createElement('div');
    s.className = 'ff-spark';
    s.style.cssText = [
      'width:'    + size               + 'px',
      'height:'   + (size * 1.5)       + 'px',
      'left:'     + xPct               + '%',
      'bottom:calc(100% + 1px)',
      'background:radial-gradient(ellipse at 50% 65%,' + c[0] + ',' + c[1] + ')',
      'box-shadow:0 0 ' + (size * 1.3).toFixed(1) + 'px ' + c[0],
      '--dx:'  + dx.toFixed(1)  + 'px',
      '--dur:' + Math.round(dur) + 'ms',
      'animation-delay:' + Math.round(delay) + 'ms',
    ].join(';');

    wrap.appendChild(s);
    setTimeout(function () {
      if (s.parentNode) s.parentNode.removeChild(s);
    }, delay + dur + 150);
  }

  function ignite(doc) {
    var wrap = doc.querySelector('div[data-testid="stChatInput"]');
    if (!wrap) return;

    if (doc.defaultView.getComputedStyle(wrap).position === 'static') {
      wrap.style.position = 'relative';
    }

    // Infinite border glow animation
    wrap.classList.remove('ff-on-fire');
    wrap.classList.add('ff-fire-infinite');

    // Cancel any previous loop
    if (window.parent._ffFireInt) {
      clearInterval(window.parent._ffFireInt);
    }

    // Initial burst of sparks
    for (var i = 0; i < 7; i++) spawnSpark(wrap, doc);

    // Continuous loop — 2–3 new sparks every ~500 ms
    window.parent._ffFireInt = setInterval(function () {
      var w = doc.querySelector('div[data-testid="stChatInput"]');
      if (!w) { clearInterval(window.parent._ffFireInt); return; }
      var n = 2 + Math.floor(Math.random() * 2);
      for (var j = 0; j < n; j++) spawnSpark(w, doc);
    }, 500);
  }

  try {
    ignite(window.parent.document);
  } catch (e) { /* cross-origin guard */ }
})();
</script>
</body></html>"""
