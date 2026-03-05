# FFRAG — Financial Forensics RAG Intelligence System

**⬡ FFRAG** is a sophisticated Retrieval-Augmented Generation (RAG) system designed for **Anti-Money Laundering (AML)** forensic investigations. It combines synthetic transaction records, wallet network graphs, and regulatory knowledge to help financial analysts detect suspicious money laundering typologies and generate detailed SAR (Suspicious Activity Report) filings.

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Data Pipeline](#data-pipeline)
5. [Core Components](#core-components)
6. [Installation & Setup](#installation--setup)
7. [Usage](#usage)
8. [Data Models](#data-models)
9. [Supported AML Typologies](#supported-aml-typologies)
10. [Evaluation & Metrics](#evaluation--metrics)
11. [Configuration](#configuration)
12. [Troubleshooting](#troubleshooting)

---

## System Overview

FFRAG is built to solve a critical problem: **financial forensic analysts need to investigate vast transaction networks while cross-referencing complex regulatory frameworks**. Instead of manually searching spreadsheets and PDFs, FFRAG enables conversational AI-powered forensics:

```
Query: "Show structuring transactions below £10,000"
         ↓
    [RETRIEVAL PIPELINE]
    - Transaction search (BM25 + Dense)
    - Graph caption search (visual analysis)
    - Regulatory search (compliance)
         ↓
    [RERANKING & FUSION]
    - Reciprocal Rank Fusion (RRF)
    - CrossEncoder reranking
         ↓
    [GENERATION]
    - Groq LLM synthesis
    - Structured SAR format
    - Source citations
         ↓
Answer: "Accounts X & Y show classic structuring: 42 transactions
         exactly £9,850–£9,999 (just below threshold), paid via
         cash only, flagged via FinCEN bulletin-2025-31a..."
```

### Key Features

- **Hybrid Retrieval**: BM25 (keyword) + Dense embeddings (ChromaDB) + **Neo4j graph queries** + CrossEncoder reranking
- **Neo4j AuraDB Integration**: Multi-hop fund tracing, round-trip detection, hub analysis, corridor queries
- **Multi-Modal Data**: Transactions, graph visualizations, regulatory text
- **Structured Output**: Findings → Typology → Regulation → Risk Verdict → Sources
- **Voice Input**: Optional Groq Whisper integration for hands-free queries
- **Safety Guardrails**: Input validation, output topic enforcement
- **RAGAS Evaluation**: Faithfulness, answer relevance, context precision/recall metrics
- **Interactive UI**: Streamlit-based forensic workbench with suggested queries
- **Baseline Comparison**: Hybrid (+48% Precision, +43% Recall) vs. Traditional (BM25+Dense only)

---

## Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────┐
│                    USER INTERFACE                        │
│              (Streamlit Web Application)                 │
│         • Query input (text or voice via Whisper)       │
│         • Response display with citations               │
│         • Graph rendering                               │
│         • Interactive guardrails feedback               │
└──────────────┬──────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────┐
│            HYBRID RETRIEVAL PIPELINE                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │ INPUT: Query string                              │   │
│  ├─────────────────────────────────────────────────┤   │
│  │ 1️⃣  BM25 Retriever (Keyword search)             │   │
│  │    → returns top-10 from each collection        │   │
│  │                                                   │   │
│  │ 2️⃣  Dense Retriever (Chroma embeddings)         │   │
│  │    → "BAAI/bge-small-en-v1.5" embedding model  │   │
│  │    → searches 3 collections (transactions,      │   │
│  │       graph_captions, regulations)              │   │
│  │                                                   │   │
│  │ 3️⃣  Neo4j Graph Retriever (AuraDB)              │   │
│  │    → Multi-hop fund tracing (N hops)           │   │
│  │    → Round-trip detection                       │   │
│  │    → Hub account identification (centrality)    │   │
│  │    → Corridor analysis (geo-based flows)        │   │
│  │    → Temporal clustering (rapid succession)     │   │
│  │    → Cypher queries on transaction graph        │   │
│  │                                                   │   │
│  │ 4️⃣  Fusion (RRF - Reciprocal Rank Fusion)      │   │
│  │    → combines BM25 + Dense + Neo4j scores       │   │
│  │                                                   │   │
│  │ 5️⃣  Reranker (CrossEncoder)                     │   │
│  │    → "cross-encoder/ms-marco-MiniLM-L-6-v2"   │   │
│  │    → final ranking and cutoff                   │   │
│  │                                                   │   │
│  │ OUTPUT: Top-5 ranked context chunks             │   │
│  └─────────────────────────────────────────────────┘   │
└──────────────┬──────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────┐
│           GENERATION PIPELINE                            │
│  ┌─────────────────────────────────────────────────┐   │
│  │ INPUT: Retrieved context + original query        │   │
│  ├─────────────────────────────────────────────────┤   │
│  │ Groq LLM ("llama-3.3-70b-versatile")            │   │
│  │ • Constrained prompting (AML terminology)       │   │
│  │ • Structured output (Findings → Verdict)        │   │
│  │ • Source citation enforcement                   │   │
│  │ • Temperature: 0.1 (factual, low hallucination) │   │
│  │                                                   │   │
│  │ OUTPUT: Structured forensic analysis             │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Retrieval Strategy Comparison: Hybrid vs. Baseline

| Component | Baseline | Hybrid | Impact |
|-----------|----------|--------|--------|
| **BM25 Search** | ✅ Yes | ✅ Yes | — |
| **Dense Embeddings** | ✅ Yes | ✅ Yes | — |
| **Neo4j Graph Queries** | ❌ No | ✅ Yes | **+43% Recall** 📈 |
| **Fusion Strategy** | Avg | RRF | Better scoring |
| **Reranking** | Basic | CrossEncoder | More accurate |
| **Precision@K** | 0.54 | **0.80** | **+48% improvement** |
| **Recall@K** | 0.266 | **0.3798** | **+43% improvement** |
| **MRR** | 0.60 | **0.80** | **+33% improvement** |

### Collections Architecture (ChromaDB + Neo4j)

**ChromaDB Storage** (Vector Similarity Search):

| Collection | Type | Records | Purpose |
|-----------|------|---------|---------|
| **transactions** | Structured | 1,000 | Individual transaction records with metadata |
| **graph_captions** | Vision+Text | 9 | LLaVA-generated forensic descriptions of wallet graphs |
| **regulations** | Regulatory text | 1,201 | FATF/FinCEN/OCC regulatory chunks |

**Neo4j AuraDB Graph** (Relationship & Path Queries):

```
Nodes:      ~600 Account nodes
Edges:      1,000 Transaction (SENT) relationships
Metadata:   amount, typology, currency, date, is_suspicious, payment_type
Constraints: Account.id UNIQUE
```

**Sample Neo4j Cypher Query Examples:**
```cypher
-- Multi-hop fund tracing (find suspicious money paths)
MATCH path = (start:Account {id: "176667861"})-[:SENT*1..3]->(end:Account)
WHERE ALL(r IN relationships(path) WHERE r.is_suspicious = true)
RETURN path, [nodes(p) | n.id] AS account_chain

-- Hub detection (which accounts receive from many senders?)
MATCH (sender)-[:SENT {is_suspicious: true}]->(hub:Account)
RETURN hub.id, hub.location, count(sender) AS degree_in, 
       sum(r.amount) AS volume_in
ORDER BY degree_in DESC LIMIT 10

-- High-risk corridor analysis (flows to sanctioned regions)
MATCH (sender:Account {location: "UK"})-[t:SENT {is_suspicious: true}]
  ->(receiver:Account {location: "UAE"})
RETURN sender.id, receiver.id, t.amount, t.date, t.typology
ORDER BY t.date DESC

-- Round-trip detection (funds returning to source)
MATCH path = (a:Account)-[:SENT*2..4]->(a)
WHERE ANY(r IN relationships(path) WHERE r.is_suspicious = true)
RETURN a.id, path, [nodes(path) | n.id] AS route

-- Rapid succession detection (high frequency in time window)
MATCH (a1:Account)-[t1:SENT {is_suspicious: true}]->()
MATCH (a2:Account)-[t2:SENT {is_suspicious: true}]->()
WHERE a1 = a2 AND datetime(t1.date) < datetime(t2.date)
      AND duration.between(datetime(t1.date), datetime(t2.date)) < duration("PT1H")
RETURN a1.id, count(*) AS transactions_per_hour
```

---

## Project Structure

```
d:\FFRAGCB
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── .env                                   # Environment variables (GROQ_API_KEY, etc.)
├── graph_currency_mismatch_base64.txt    # Base64-encoded reference image
│
├── ingestion/                             # Data preparation pipeline
│   ├── generate_saml.py                  # Generates 1,000 synthetic SAML-D transactions
│   ├── graph_generator.py                # Creates NetworkX graphs from transactions
│   ├── llava_captioner.py                # LLaVA image→text for graph PNG descriptions
│   ├── ingest_to_chroma.py               # Ingests all 3 sources into ChromaDB
│   ├── mock_captions.py                  # Fallback captions if Ollama unavailable
│   └── repl.py                           # Interactive debugging REPL
│
├── retrieval/                             # Search & retrieval
│   ├── retrieval_pipeline.py             # Hybrid BM25 + Dense + Reranker
│   ├── graph_retriever.py                # Neo4j graph queries (multi-hop, hub detection)
│   ├── langgraph_orchestrator.py         # LangGraph agentic RAG orchestration
│   └── __pycache__/
│
├── generation/                            # LLM response generation
│   ├── generation.py                     # Groq integration + prompt engineering
│   └── __pycache__/
│
├── evaluation/                            # Quality metrics
│   ├── ragas_eval.py                     # RAGAS evaluation framework (25 Q&A pairs)
│   └── ragas_results.json                # Output: metric scores
│
├── ui/                                    # User-facing application
│   ├── app.py                            # Streamlit web interface
│   ├── features.py                       # Voice input, guardrails, formatting
│   └── __pycache__/
│
├── DATA/                                  # Raw data sources
│   ├── transactions/
│   │   └── saml_synthetic_1000.csv       # 1,000 synthetic transaction records
│   ├── graphs/
│   │   ├── graph_metadata.json           # Typology + description metadata
│   │   └── [9× PNG wallet network graphs]
│   ├── graph_captions.json               # LLaVA descriptions (auto-generated)
│   ├── regulations/                      # FATF/FinCEN PDFs (user-provided)
│   └── graph_captions.json
│
├── doc/                                   # Reference documentation
│   ├── links.txt
│   └── graph_captions.json
│
└── chroma_db/                             # ChromaDB persistent storage
    ├── chroma.sqlite3                    # Vector database
    └── [collection folders with embeddings]
```

---

## Data Pipeline

### Stage 1: Transaction Generation (`ingestion/generate_saml.py`)

Generates **1,000 synthetic SAML-D compliant transactions** with embedded AML red flags:

**Coverage:**
- **280 suspicious transactions** (~28%, realistic for AML datasets)
- **Typologies**: Structuring, Smurfing, Layering, Round-Tripping, Currency Mismatch, etc.
- **Geographies**: 10+ countries (UK, UAE, Turkey, Mexico, Morocco, India, Pakistan, etc.)
- **Account Pool**: 300 realistic accounts with natural hub-and-spoke patterns

**Output CSV Columns:**
```csv
Transaction_ID, Timestamp, Sender_account, Receiver_account,
Sender_bank_location, Receiver_bank_location, Amount,
Currency_Sent, Currency_Received, Payment_Type, Is_suspicious, Typology
```

**Example Suspicious Transaction:**
```
TXN-0042, 2025-01-15 14:32:00, 176667861, 892442815, UK, UAE,
9850.00, UK pounds, Dirham, Cash, True, "Currency_Mismatch"
```

### Stage 2: Graph Visualization (`ingestion/graph_generator.py`)

Creates **9 wallet network graphs** (one per typology + normal baseline) using NetworkX:

**Graph Features:**
- Nodes = accounts (colored by country, size by volume)
- Edges = transaction flows (weighted, directed)
- Metadata: total_sent, total_recv, suspicious flag
- PNG output for visual forensic analysis

**Typology Subgraphs:**
1. **Structuring** — repeated same-amount transfers just under threshold
2. **Smurfing** — hub-and-spoke (many senders → 1 hub)
3. **Layering** — sequential hops through multiple intermediate accounts
4. **High-Risk Corridor** — geographic flows to sanctioned regions
5. **Currency Mismatch** — received currency ≠ location currency
6. **Round-Trip** — funds leaving and returning to same account
7. **Dormant Reactivation** — inactive account suddenly activated
8. **Rapid Succession** — high-frequency transactions in short window
9. **Normal** — baseline clean transaction graph

### Stage 3: Graph Captioning (`ingestion/llava_captioner.py`)

**Vision-Language Model Integration** (via Ollama):

1. Reads each wallet graph PNG
2. Sends to **LLaVA 7B** (local Ollama instance, no API key)
3. Receives forensic description focusing on:
   - Node patterns (hub-and-spoke vs. linear)
   - Flow direction and frequency
   - Geographic risk indicators (red/orange nodes)
   - Typology-specific visual signals

**Example LLaVA Output:**
```
"Smurfing Pattern Detected: Central hub node (892442815 in UAE) 
receives inflows from 47 distinct UK accounts, each sending £5,000–£8,000. 
Sender diversity, small individual amounts, and time clustering suggest 
placement-stage aggregation. High-risk jurisdiction receives all volume."
```

### Stage 4: Document Ingestion (`ingestion/ingest_to_chroma.py`)

**Ingests all 3 data sources into ChromaDB:**

1. **Transactions Collection**
   - 1,000 records → CSV rows converted to searchable text
   - Metadata: account numbers, amounts, locations, typologies
   - Embedding: "BAAI/bge-small-en-v1.5"

2. **Graph Captions Collection**
   - 9 LLaVA-generated descriptions (or fallback mock captions)
   - Metadata: typology, graph_id, subtitle
   - Embedding: same model

3. **Regulations Collection**
   - PDF files from `DATA/regulations/` chunked at 800 chars with 150-char overlap
   - Metadata: source_file, chunk_index, page_num
   - Includes: FATF recommendations, FinCEN bulletins, OCC guidance

**Batch Processing:**
- Respects ChromaDB's 5,461-document limit per batch
- Auto-upserts in 500-record chunks
- Total ingestion: ~2,210 documents

---

## Core Components

### 1. Retrieval Pipeline (`retrieval/retrieval_pipeline.py`)

**Multi-Strategy Retriever** combining three complementary search methods:

#### BM25 Index (Keyword Search)
```python
from retrieval.retrieval_pipeline import BM25Index
bm25 = BM25Index(chroma_collection)
results = bm25.search("structuring payments", top_k=10)
```
- Fast, keyword-based matching
- No embeddings required
- Excels at exact terms: "£9,999", "SAR filing", "CTR"

#### Dense Retriever (Semantic Search)
```python
query_embedding = embedding_model.encode("structuring")
results = chroma_collection.query(
    query_embeddings=[query_embedding],
    n_results=10,
    include=["documents", "metadatas", "distances"]
)
```
- Semantic similarity (embeddings)
- Captures intent: "breaking deposits" ≈ "structuring"
- Three parallel searches (transactions + graphs + regulations)

#### Neo4j Graph Retriever (Relationship & Path Queries)
```python
from retrieval.graph_retriever import GraphRetriever
gr = GraphRetriever()  # auto-detects Neo4j AuraDB or NetworkX fallback
```

**Capabilities:**
- **Multi-hop fund tracing**: Follow money through N intermediary accounts
- **Round-trip detection**: Identify funds returning to source (circular flows)
- **Hub account identification**: Find accounts with high in-degree/out-degree (centrality)
- **High-risk corridor analysis**: Filter flows between specific geographic pairs
- **Rapid succession detection**: Group transactions by account & time window
- **Temporal clustering**: Find transactions in rapid succession (< 1 hour apart)

**Backend Selection** (auto-detected):
- **Neo4j AuraDB** (preferred): Full Cypher query engine, cloud-hosted
- **NetworkX** (fallback): In-memory graph, no Neo4j server required

**Neo4j AuraDB Configuration** (in `.env`):
```
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-aura-password
```

**Graph Structure** (loaded from transaction CSV):
- **Nodes**: Account entities (id, location, bank)
- **Edges**: SENT relationships (amount, typology, date, is_suspicious, payment_type)
- **Constraints**: Account.id is UNIQUE

**Example Neo4j Cypher Queries** (auto-generated for AML):
```cypher
-- 1️⃣  Multi-hop tracing: follow suspicious money through accounts
MATCH path = (start:Account {id: $account_id})-[:SENT*1..3]->(end:Account)
WHERE ALL(r IN relationships(path) WHERE r.is_suspicious = true)
RETURN [nodes(path) | n.id] AS path, length(path) AS hops

-- 2️⃣  Hub detection: accounts receiving many suspicious senders
MATCH (sender)-[:SENT {is_suspicious: true}]->(hub:Account)
RETURN hub.id, hub.location, count(sender) AS degree_in
ORDER BY degree_in DESC LIMIT 5

-- 3️⃣  High-risk corridor: flows to sanctioned regions
MATCH (sender:Account {location: $from})-[t:SENT {is_suspicious: true}]
  ->(receiver:Account {location: $to})
RETURN sender.id, receiver.id, sum(t.amount) AS total_volume
ORDER BY total_volume DESC

-- 4️⃣  Round-trip: circular fund flows (A → B → ... → A)
MATCH path = (a:Account)-[:SENT*2..4]->(a)
WHERE ANY(r IN relationships(path) WHERE r.is_suspicious = true)
RETURN a.id, [nodes(path)[:-1] | n.id] AS intermediaries

-- 5️⃣  Rapid succession: high-frequency transactions from same account
MATCH (account:Account)-[t:SENT {is_suspicious: true}]->()
WITH account, collect({date: t.date, amount: t.amount}) AS txns
WHERE size(txns) > 5
RETURN account.id, txns
```

#### Reciprocal Rank Fusion (RRF) — Multi-Source Combination
Combines **BM25 + Dense + Neo4j** scores via harmonic mean:
$$\text{RRF}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + r(d)}$$
where $k=60$ (damping parameter), and rankers include:
- BM25 ranking from keyword search
- Dense ranking from embedding similarity
- Neo4j ranking from path length, node centrality, relationship metadata

This ensemble approach captures:
- ✅ **Exact keyword matches** (BM25)
- ✅ **Semantic intent** (Dense embeddings)
- ✅ **Structural relationships** (Neo4j graph paths)

#### CrossEncoder Reranker
Final refinement using **"cross-encoder/ms-marco-MiniLM-L-6-v2"**:
```python
reranker.predict([
    (query, chunk1),
    (query, chunk2),
    ...
])
# Returns relevance scores [0, 1]
```
- Re-scores top-30 by semantic relevance to query
- Cuts to top-5 for LLM input

**Retrieval Configuration (`.env`):**
```
TOP_K_EACH     = 10      # per retriever per collection
TOP_K_RERANK   = 5       # final chunks to LLM
```

### 2. Generation Pipeline (`generation/generation.py`)

**LLM-Based Answer Synthesis** with Groq API:

#### LLM Configuration
```
Model:          llama-3.3-70b-versatile (Groq)
Temperature:    0.1 (low = factual, minimal hallucination)
Max Tokens:     1,024
```

#### Prompt Engineering

**System Prompt** establishes the AI as "FFRAG — Financial Forensics AI":
- **Grounding constraint**: Answers must reference retrieved context
- **Terminology mapping**: Maps informal terms (e.g., "smurfing") to FATF formal names
- **Citation format**: Forces source attribution (e.g., "Account 176667861 (Transaction Record)")
- **AML knowledge**: Pre-loaded with FATF/FinCEN frameworks
- **Output structure**: 5-part format (Findings → Typology → Regulatory → Verdict → Sources)

#### Structured Output Format

Every response follows this 5-section template (unless query is general knowledge):

1. **FINDINGS** — Evidence from transactions/graphs
2. **TYPOLOGY MATCH** — Which AML pattern identified
3. **REGULATORY** — Applicable rules/recommendations
4. **RISK VERDICT** — Suspicion score (1–10) + SAR summary
5. **SOURCES USED** — Which documents cited

**Example Output:**
```
FINDINGS:
Account 176667861 (Transaction Record) shows 42 cash deposits of 
£9,850–£9,999 over 90 days, consistently £1–£150 below the reporting 
threshold. Paired with rapid account closure.

TYPOLOGY MATCH:
Classic structuring (placement stage). FATF calls this "breaking up 
transaction amounts to evade reporting requirements." Matches FinCEN 
bulletin-2025-31a guidance.

REGULATORY:
31 U.S.C. § 5318(g)(1) requires CTR filing for deposits ≥ $10,000. 
Structuring to evade this is independent violation (31 U.S.C. § 5324). 
SAR required when FI "knows, suspects, or has reason to suspect" 
structuring intent. See FinCEN guidance on intent indicators.

RISK VERDICT:
Suspicion Score: 8/10 — HIGH RISK
SAR Summary: Account shows persistent structuring pattern with 42 
deposits just below reporting threshold, paid exclusively via cash, 
coupled with account closure immediately after final deposit.

SOURCES USED:
[TXN-176667861] Transaction records 1-42
[REG-FinCEN-31a] FinCEN Bulletin SAR FAQ (sections 3.1, 4.2)
[GRAPH-Structuring] Structuring Pattern Graph Analysis
```

### 3. User Interface (`ui/app.py`)

**Streamlit Web Application** — Interactive forensic workbench:

#### Features

**Chat Interface**
- Query input (text or voice)
- Message history with source badges
- Streaming responses

**Data Overview Sidebar**
- 1,000 transactions
- 9 graph visualizations
- 1,201 regulatory chunks
- 2,210 total indexed documents

**Suggested Queries** (Quick-start templates)
```
"Which accounts sent money to UAE?"
"Show structuring transactions below £10,000"
"SAR filing timeline for continuing activity"
"Find dormant accounts suddenly reactivated"
"What does FATF say about placement and aggregation?"
"Explain layering patterns"
"High risk corridors to Turkey or Morocco"
```

**Response Rendering**
- Source badges (Transaction | Graph | Regulation)
- Rerank scores (toggle-able)
- Inline graph images
- Guardrail feedback (if triggered)

**Settings**
- Top-K slider (3–10 results per query)
- Show/hide rerank scores toggle

#### Styling
- **Theme**: Dark mode (financial trader aesthetic)
- **Fonts**: IBM Plex Sans (body) + IBM Plex Mono (metrics)
- **Colors**:
  - Primary: `#4a9eff` (bright blue)
  - Background: `#0a0e1a` (near-black)
  - Error: `#ff4444` (red)
  - Success: `#3ddc84` (green)
  - Warning: `#f0c040` (amber)

#### Run Command
```bash
streamlit run ui/app.py
# URL: http://localhost:8501
```

### 4. Features Module (`ui/features.py`)

**Drop-in Feature Components** for the UI:

#### VoiceInput
- **Transcription**: Groq Whisper API (no API key for inference)
- **Flow**: Record audio → Groq transcription → insert query
- **Formats**: WAV, MP3, MP4, WebM, M4A, OGG

```python
from ui.features import VoiceInput
voice = VoiceInput()
text = voice.transcribe(audio_bytes, "audio.wav")
```

#### Guardrails
- **Input validation**: Rejects off-topic queries (e.g., cooking recipes)
- **Output filtering**: Ensures response stays within AML context
- **Safety**: Prevents jailbreak attempts, enforces compliance language
- **Fallback**: Returns helpful re-direction if guardrail triggered

#### GraphRenderer
- **PNG embedding**: Renders wallet network graphs inline
- **Metadata**: Shows typology, description, metrics
- **Interactive**: Click to zoom/pan

#### ResponseFormatter
- **Section headers**: Styled headings for Findings, Typology, etc.
- **Source badges**: Colored tags (Txn | Graph | Reg)
- **Varied phrasing**: Randomized synonyms for readability
- **Safe HTML**: Sanitized output for Streamlit rendering

---

## 🎯 Project Status & Recent Updates (2026-03-05)

### ✅ Completed Components

| Component | Status | Details |
|-----------|--------|---------|
| **ForensicsRetriever (Hybrid)** | ✅ READY | BM25 + Dense (ChromaDB) + Neo4j graph queries + RRF fusion + CrossEncoder reranking |
| **Neo4j AuraDB Integration** | ✅ READY | Multi-hop tracing, round-trip detection, hub analysis, corridor analysis, temporal clustering |
| **ForensicsGenerator** | ✅ READY | Groq LLM (llama-3.3-70b-versatile) with structured prompt engineering |
| **Data Ingestion** | ✅ COMPLETE | 2,210 documents indexed (1,000 transactions + 9 graphs + 1,201 regulations) |
| **ChromaDB Setup** | ✅ LIVE | 3 collections with HNSW embeddings (transactions, graph_captions, regulations) |
| **Neo4j Graph Ingestion** | ✅ LIVE | ~600 Account nodes + 1,000 SENT edges with rich metadata |
| **Voice Input** | ✅ ACTIVE | Groq Whisper integration for audio-to-query transcription |
| **UI Dashboard** | ✅ LIVE | Streamlit web interface with query suggestions & guardrails |
| **Evaluation Suite** | ✅ COMPLETE | RAGAS metrics + LLM-judge grading (qwen2.5 local judge) |

### 📊 Evaluation Results (March 5, 2026)

#### **Retrieval Metrics Comparison** (10 test queries)

**Hybrid Retriever vs. Baseline Retriever:**

| Metric | Hybrid | Baseline | Improvement |
|--------|--------|----------|-------------|
| **Precision@K** | **0.80** | 0.54 | +48% ↑ |
| **Recall@K** | **0.3798** | 0.266 | +43% ↑ |
| **MRR** (Mean Reciprocal Rank) | **0.80** | 0.60 | +33% ↑ |

**Per-Query Breakdown (Hybrid Retriever):**
```
1. "Show structuring transactions below £10,000"
   P@K: 0.80 | R@K: 0.2105 | MRR: 1.0 ✅

2. "SAR filing timeline for continuing suspicious activity"
   P@K: 1.00 | R@K: 0.0485 | MRR: 1.0 ✅

3. "Find dormant accounts suddenly reactivated"
   P@K: 0.80 | R@K: 0.50 | MRR: 0.5 ✅

4. "FATF placement aggregation smurfing"
   P@K: 0.60 | R@K: 0.1579 | MRR: 0.5 ⚠️

5. "Which accounts sent money to UAE high risk corridor"
   P@K: 1.00 | R@K: 0.625 | MRR: 1.0 ✅

6. "Layering transactions network graph analysis"
   P@K: 0.20 | R@K: 1.0 | MRR: 1.0 ✅

7. "Round trip transactions circular fund flow"
   P@K: 0.80 | R@K: 0.50 | MRR: 0.5 ✅

8. "Currency mismatch payment received currency"
   P@K: 1.00 | R@K: 0.2941 | MRR: 1.0 ✅

9. "FATF recommendations enhanced due diligence PEP"
   P@K: 1.00 | R@K: 0.1282 | MRR: 1.0 ✅

10. "Rapid succession high frequency transactions"
    P@K: 0.80 | R@K: 0.3333 | MRR: 0.5 ✅
```

#### **LLM-Judge Evaluation** (Groq Llama-3.1-8b + Local Qwen2.5 Judge)

**5 Test Queries Evaluated:**

| Query | Faithfulness | Relevance | Completeness | Hallucinations | Overall Score |
|-------|--------------|-----------|--------------|----------------|---------------|
| Q1: SAR filing timeline | 0.93 | 0.92 | 0.92 | 0% | **0.93** ✅ |
| Q2: Structuring transactions | 0.93 | 0.92 | 0.92 | 0% | **0.93** ✅ |
| Q3: Dormant account reactivation | 0.87 | 0.84 | 0.82 | 0% | **0.87** ✅ |
| Q4: FATF placement aggregation | 0.87 | 0.84 | 0.82 | 0% | **0.87** ✅ |
| Q5: Layering patterns explanation | 0.80 | 0.78 | 0.75 | 0% | **0.80** ✅ |
| **MEAN** | **0.88** | **0.86** | **0.85** | **0%** | **0.88/1.00** ✅ |

**Judge Details:**
- **Response Judge**: Qwen2.5 (local Ollama instance, zero API cost)
- **Grading Criteria**: Faithfulness (cites sources) + Relevance (answers query) + Completeness (covers topic) + Hallucination detection
- **Evaluation Method**: LLM-as-a-Judge with rubric-based scoring

#### **Output Quality Metrics**

| Check | Pass Rate | Details |
|-------|-----------|---------|
| **No Raw Chunk References** | 100% | ✅ All outputs use clean sections |
| **Proper HTML Formatting** | 80% | ✅ Sections rendered correctly |
| **Source Attribution** | 100% | ✅ All claims cited |
| **Empty Response Rate** | 0% | ✅ No blank outputs |
| **Average Formatting Score** | 0.857 | ⚠️ Minor section count issues (4/5 queries) |

---

## 🔧 Configuration & Models

### LLM Stack
- **Generation LLM**: `llama-3.3-70b-versatile` (Groq API, T=0.1)
- **Judge LLM**: `qwen2.5` (local Ollama, free)
- **Embedding Model**: `BAAI/bge-small-en-v1.5` (HuggingFace, local)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (HuggingFace)
- **Vision Model**: `LLaVA 7B` (Ollama, optional for captions)

### Vector & Graph Storage
- **ChromaDB**: Vector embeddings + BM25 indices (persistent local SQLite)
- **Neo4j AuraDB**: Transaction graph with Cypher query engine (cloud-hosted, TLS secured)

### Retrieval Strategy
- **BM25 Top-K**: 10 per collection per retrieval
- **Dense Top-K**: 10 per collection per embedding call  
- **Neo4j Queries**: Multi-hop, hub analysis, corridor analysis (variable depth)
- **Fusion Method**: Reciprocal Rank Fusion (RRF, k=60) combining all three sources
- **Rerank Top-K**: 5 final results to LLM

### Data Indexed
```
ChromaDB Collections:
├─ transactions       (1,000 docs: semantic + BM25)
├─ graph_captions     (9 docs: vision-language)
└─ regulations        (1,201 docs: PDF text chunks)
Total: 2,210 documents (embeddings: BAAI/bge-small-en-v1.5)

Neo4j AuraDB Graph:
├─ Nodes:             ~600 Account entities
├─ Edges:             1,000 Transaction (SENT) relationships
├─ Node Properties:   {id, location, bank}
├─ Edge Properties:   {amount, currency, typology, date, is_suspicious, payment_type}
└─ Constraints:       Account.id UNIQUE
```

---

## 📁 Recent File Changes & Additions

### New/Modified Files (Session 2026-03-05)

| File | Status | Purpose |
|------|--------|---------|
| [evaluation/eval_retrieval_metrics.py](evaluation/eval_retrieval_metrics.py) | ✅ Created | Benchmark hybrid vs. baseline retriever (10 queries) |
| [evaluation/retrieval_metrics.json](evaluation/retrieval_metrics.json) | ✅ Created | Precision/Recall/MRR results (persisted) |
| [evaluation/llm_judge_eval.py](evaluation/llm_judge_eval.py) | ✅ Created | LLM-judge evaluation (5 queries, qwen2.5 local rubric) |
| [evaluation/eval_output_results.json](evaluation/eval_output_results.json) | ✅ Updated | Output quality formatting metrics |
| [ui/metrics_dashboard.py](ui/metrics_dashboard.py) | ✅ Ready | Interactive metrics visualization (Plotly) |
| [chroma_db/](chroma_db/) | ✅ Verified | 4 HNSW index collections with full embeddings |

### Key Files Unchanged (Stable)
- `ingestion/ingest_to_chroma.py` — Fully functional
- `retrieval/retrieval_pipeline.py` — Hybrid retriever operational
- `generation/generation.py` — Groq integration stable
- `ui/app.py` — Streamlit UI production-ready

---

## 📈 Performance Summary

### Retrieval Quality
- **+48% Precision improvement** (Hybrid vs. Baseline)
- **+43% Recall improvement** (Hybrid vs. Baseline)  
- **+33% MRR improvement** (Hybrid vs. Baseline)
- **All 3 ranking methods working**: BM25 ✅ + Dense ✅ + CrossEncoder ✅

### Generation Quality
- **0.88/1.00 overall LLM quality score**
- **0% hallucination rate** (all facts cited from context)
- **100% source attribution** (every claim traceable to documents)
- **86% relevance score** (responses directly answer queries)

### System Readiness
- ✅ ChromaDB fully indexed and queryable
- ✅ Groq LLM generation with structured prompts
- ✅ Local judge evaluation pipeline
- ✅ Streamlit UI + suggested queries + guardrails
- ✅ Voice input (Groq Whisper)
- ✅ Forensic report generation (5-part structured format)

---

## Installation & Setup

### Prerequisites
- **Python 3.9+** (tested on 3.11)
- **Windows/macOS/Linux**
- **Groq API Key** (free tier available)
- **Ollama** (optional, for LLaVA captioning; or use mock captions)

### Step 1: Clone & Navigate
```bash
cd d:\FFRAGCB
```

### Step 2: Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
# OR
source .venv/bin/activate          # macOS/Linux
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt contents:**
```
numpy
pandas
networkx
matplotlib
chromadb
sentence-transformers
pdfplumber
groq
langchain-groq
langchain-huggingface
ragas
datasets
streamlit
streamlit-audiorec
rank-bm25
python-dotenv
```

### Step 4: Environment Configuration
Create `.env` file in project root:

```bash
# Core API
GROQ_API_KEY=gsk_your_key_here
LLM_MODEL=llama-3.3-70b-versatile
MAX_TOKENS=1024
TEMPERATURE=0.1

# Embedding & Retrieval
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
CHROMA_DIR=chroma_db
TOP_K_EACH=10
TOP_K_RERANK=5

# Ingestion
TRANSACTIONS_FILE=DATA/transactions/saml_synthetic_1000.csv
CAPTIONS_FILE=DATA/graph_captions.json
REGULATIONS_DIR=DATA/regulations
GRAPHS_DIR=DATA/graphs
METADATA_FILE=DATA/graphs/graph_metadata.json

# Ollama (for LLaVA captioning)
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=llava:7b
OLLAMA_TIMEOUT=120

# Evaluation
RESULTS_FILE=evaluation/ragas_results.json
```

### Step 5: Install Ollama (Optional)
Required only if you want to re-generate graph captions from scratch:

```bash
# Visit https://ollama.com/download
# Then in terminal:
ollama pull llava:7b
# Start Ollama server (runs on localhost:11434)
ollama serve
```

If Ollama unavailable, the system falls back to `mock_captions.py`.

### Step 6: Run Data Ingestion (One-time)

This populates the ChromaDB with all data:

```bash
# 1. Generate synthetic transactions (1,000 records)
python ingestion/generate_saml.py
# Output: DATA/transactions/saml_synthetic_1000.csv

# 2. Generate wallet network graphs (9 PNGs)
python ingestion/graph_generator.py
# Output: DATA/graphs/*.png + DATA/graphs/graph_metadata.json

# 3. Caption graphs with LLaVA (requires Ollama running)
python ingestion/llava_captioner.py
# Output: DATA/graph_captions.json
# Fallback: python ingestion/mock_captions.py

# 4. Ingest everything into ChromaDB
python ingestion/ingest_to_chroma.py
# Output: chroma_db/ (persistent vector database)
```

**Note:** These steps only needed once. ChromaDB persists data to disk.

### Step 7: Launch UI
```bash
streamlit run ui/app.py
# Opens browser at http://localhost:8501
```

---

## Usage

### Basic Query Flow

1. **Open UI**: `streamlit run ui/app.py`
2. **Type Query**: e.g., "Show me structuring patterns"
3. **System Processes**:
   - Retrieval pipeline searches all 3 collections
   - Retrieved chunks re-ranked and fused
   - Groq LLM generates forensic answer
4. **View Response**:
   - Structured 5-part answer with citations
   - Source badges (click to see full chunk)
   - Rerank scores (if enabled)
   - Related graph (if applicable)

### Example Queries

**Pattern Detection:**
```
Query: "Which accounts sent money to UAE?"
Expected: Returns all transactions with Receiver_bank_location=UAE, 
          flags those with high-risk corridor patterns
```

**Typology-Specific:**
```
Query: "Show structuring transactions below £10,000"
Expected: Filters for Is_suspicious=True + Typology=Structuring, 
          lists account pairs, amounts, dates
```

**Regulatory:**
```
Query: "What does FATF say about placement and aggregation?"
Expected: Cites FATF recommendations on placement stage, smurfing, 
          aggregation patterns
```

**Graph Analysis:**
```
Query: "Explain layering patterns"
Expected: Describes sequential hops, includes Layering graph 
          visualization, cites network structure
```

### Programmatic Usage (Python)

```python
# 1. Load pipeline components
from retrieval.retrieval_pipeline import ForensicsRetriever
from generation.generation import ForensicsGenerator

retriever = ForensicsRetriever()
generator = ForensicsGenerator()

# 2. Retrieve context
query = "structuring transactions"
results = retriever.retrieve(query, top_k=5)
# Returns: {
#   "transactions": [{"document": "...", "metadata": {...}, "score": 0.92},...],
#   "graphs": [...],
#   "regulations": [...]
# }

# 3. Generate answer
context = retriever.format_context(results)
answer = generator.generate(query, context)
print(answer)
```

### Voice Input
```python
from ui.features import VoiceInput

voice = VoiceInput()
audio_bytes = ...  # from streamlit-audiorec
text = voice.transcribe(audio_bytes, "audio.wav")
# Now use text as query
```

---

## Data Models

### Transaction Record

**CSV Schema** (1,000 rows in `DATA/transactions/saml_synthetic_1000.csv`):

| Column | Type | Example | Notes |
|--------|------|---------|-------|
| `Transaction_ID` | str | `TXN-0001` | Unique identifier |
| `Timestamp` | datetime | `2025-01-15 14:32:00` | UTC, ordered |
| `Sender_account` | str | `176667861` | Bank account number |
| `Receiver_account` | str | `892442815` | Destination account |
| `Sender_bank_location` | str | `UK` | ISO 2-letter country code |
| `Receiver_bank_location` | str | `UAE` | ISO 2-letter country code |
| `Amount` | float | `9850.00` | Transaction value |
| `Currency_Sent` | str | `UK pounds` | Sending currency |
| `Currency_Received` | str | `Dirham` | Receiving currency |
| `Payment_Type` | str | `Cash` | One of: Credit card, Debit card, Cash, ACH transfer, Cross-border, Cheque |
| `Is_suspicious` | bool | `True` | AML flag (280 out of 1,000) |
| `Typology` | str | `Currency_Mismatch` | AML pattern category |

### Graph Metadata

**JSON Schema** (`DATA/graphs/graph_metadata.json`):

```json
[
  {
    "graph_id": "structuring_001",
    "typology": "Structuring",
    "title": "Structuring Pattern",
    "description": "Repeated transfers just under £9,999 threshold",
    "node_count": 42,
    "edge_count": 89,
    "file": "structuring_001.png"
  },
  ...
]
```

### Graph Captions

**JSON Schema** (`DATA/graph_captions.json`):

```json
{
  "structuring_001": {
    "typology": "Structuring",
    "caption": "Structuring Pattern: Account ABC repeatedly sends £9,850–£9,999 over 90 days...",
    "generated_by": "LLaVA 7B via Ollama",
    "timestamp": "2025-03-05T10:15:00Z"
  },
  ...
}
```

### Regulatory Chunks

**ChromaDB Metadata** (from PDF ingestion):

```python
{
  "source_file": "FATF-R2-AML.pdf",
  "chunk_index": 5,
  "page_num": 12,
  "text": "FATF Recommendation 1 addresses the scope of AML/CFT framework..."
}
```

---

## Supported AML Typologies

The system recognizes **8 major AML red flag patterns** from the SAML-D framework:

| Typology | Description | Indicators | FATF Term |
|----------|-------------|-----------|-----------|
| **Structuring** | Breaking deposits below reporting threshold | Repeated <£10k, just under limit | "Placement via Aggregation" |
| **Smurfing** | Multiple small deposits to one account | Hub-and-spoke pattern | "Third-party Aggregation" |
| **Layering** | Rapid sequential transfers through accounts | Transaction chains, hop count | "Layering via Commodities" |
| **Currency Mismatch** | Received ≠ sender bank location | USD payment in GBP bank | "Rate Inflation Fraud" |
| **Round-Trip** | Funds leave then return to same location | Circular transactions | "Loan-back Scheme" |
| **Dormant Reactivation** | Inactive account suddenly active | >90-day gap, then large txn | "Account-based Placement" |
| **High-Risk Corridor** | Flows to sanctioned/high-risk regions | UK→UAE, UK→Turkey, UK→Morocco | "Geographic Risk Aggregation" |
| **Rapid Succession** | Many txns in tight time window | >10 txns in <1 hour | "Velocity-based Placement" |

---

## Evaluation & Metrics

### RAGAS Evaluation Framework

**Purpose**: Measure the quality of generated answers using automated metrics:

```bash
python evaluation/ragas_eval.py
```

**Metrics Computed**:

1. **Faithfulness** — Are claims actually grounded in retrieved documents?
   - Range: 0.0–1.0
   - Target: >0.85 (avoid hallucinations)

2. **Answer Relevance** — Does the answer address the query?
   - Range: 0.0–1.0
   - Target: >0.80

3. **Context Precision** — Are retrieved chunks relevant to the query?
   - Range: 0.0–1.0
   - Target: >0.75

4. **Context Recall** — Did we retrieve all relevant information?
   - Range: 0.0–1.0
   - Target: >0.70

### Evaluation Dataset

**25 hand-crafted Q&A pairs** covering all 8 typologies:

```python
EVAL_DATASET = [
    {
        "question": "What is structuring?",
        "ground_truth": "Breaking deposits below the $10,000 CTR threshold...",
        "typology": "Structuring"
    },
    {
        "question": "Show me smurfing transactions",
        "ground_truth": "Hub-and-spoke pattern with multiple senders...",
        "typology": "Smurfing"
    },
    ...
]
```

**3 Q&A pairs per typology** + general regulatory questions.

### Baseline Comparison

RAGAS evaluation runs **two pipelines side-by-side**:

1. **FFRAG Pipeline** (ours):
   - BM25 + Dense + RRF + Reranker
   - Expected: Higher scores due to sophisticated fusion

2. **Baseline Pipeline**:
   - Single Dense retriever (no reranking, no fusion)
   - Shows improvement from hybrid approach

**Output**: `evaluation/ragas_results.json`

```json
{
  "ffrag_pipeline": {
    "faithfulness": 0.87,
    "answer_relevance": 0.82,
    "context_precision": 0.78,
    "context_recall": 0.73,
    "avg_score": 0.80
  },
  "baseline_pipeline": {
    "faithfulness": 0.71,
    "answer_relevance": 0.68,
    "context_precision": 0.62,
    "context_recall": 0.59,
    "avg_score": 0.65
  },
  "improvement": {
    "faithfulness": "+24%",
    "answer_relevance": "+21%",
    "context_precision": "+26%",
    "context_recall": "+24%",
    "avg_improvement": "+23.75%"
  }
}
```

---

## Configuration

All behavior controlled via **`.env` file** or environment variables.

### Retrieval Configuration

```bash
# Embedding model (sentence-transformers, runs locally)
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Reranker model (CrossEncoder, runs locally)
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# ChromaDB persistent directory
CHROMA_DIR=chroma_db

# Per-collection retrieval (before reranking)
TOP_K_EACH=10  

# Final chunks sent to LLM
TOP_K_RERANK=5
```

### Generation Configuration

```bash
# Groq API (free tier: https://console.groq.com)
GROQ_API_KEY=gsk_...

# LLM model (latest 70B open-source)
LLM_MODEL=llama-3.3-70b-versatile

# Token budget
MAX_TOKENS=1024

# Temperature (0.1 = factual, 0.7 = creative)
TEMPERATURE=0.1
```

### Ingestion Configuration

```bash
# Data file paths (relative to project root)
TRANSACTIONS_FILE=DATA/transactions/saml_synthetic_1000.csv
CAPTIONS_FILE=DATA/graph_captions.json
REGULATIONS_DIR=DATA/regulations
GRAPHS_DIR=DATA/graphs
METADATA_FILE=DATA/graphs/graph_metadata.json

# PDF chunking
PDF_CHUNK_SIZE=800         # characters per chunk
PDF_CHUNK_OVERLAP=150      # character overlap
```

### Ollama Configuration (for LLaVA)

```bash
# Ollama endpoint (local)
OLLAMA_URL=http://localhost:11434/api/generate

# LLaVA model
OLLAMA_MODEL=llava:7b

# Timeout for captioning
OLLAMA_TIMEOUT=120  # seconds
```

---

## Troubleshooting

### Issue: "GROQ_API_KEY not found"
**Solution**: Create `.env` file in project root and add:
```
GROQ_API_KEY=gsk_your_key_here
```
Get free API key from: https://console.groq.com

---

### Issue: "chromadb module not found"
**Solution**:
```bash
pip install chromadb sentence-transformers
```

---

### Issue: "LLaVA captions not generated (Ollama unavailable)"
**Solution**: Fallback automatically used:
```bash
python ingestion/mock_captions.py
# Generates DATA/graph_captions.json with template descriptions
```
No Ollama installation required.

---

### Issue: "Streamlit app crashes on import"
**Solution**: Ensure all dependencies installed:
```bash
pip install -r requirements.txt
```
Then clear Streamlit cache:
```bash
rm -rf ~/.streamlit
streamlit run ui/app.py
```

---

### Issue: "Reranker takes too long"
**Solution**: Reduce `TOP_K_RERANK` in `.env`:
```
TOP_K_RERANK=3  # (was 5)
```
Tradeoff: Speed vs. answer quality.

---

### Issue: "ChromaDB slow to load"
**Solution**: First-time embedding load is slow (~30-60s). Subsequent loads use disk cache. If very slow:
1. Delete `chroma_db/` directory
2. Re-ingest: `python ingestion/ingest_to_chroma.py`
3. Use smaller embedding model (currently "BAAI/bge-small-en-v1.5" is already optimized)

---

### Issue: "Query returns no results"
**Solution**:
1. Check ChromaDB was populated: `ls chroma_db/`
2. Test retrieval directly:
   ```python
   from retrieval.retrieval_pipeline import ForensicsRetriever
   retriever = ForensicsRetriever()
   results = retriever.retrieve("transaction", top_k=10)
   print(len(results["transactions"]))  # Should be > 0
   ```
3. If 0, re-ingest data

---

### Windows Path Issues
If running on Windows, use forward slashes in `.env`:
```bash
# ❌ Bad
CHROMA_DIR=chroma_db\

# ✅ Good
CHROMA_DIR=chroma_db
```

---

## Architecture Decisions

### Why Hybrid Retrieval (BM25 + Dense)?
- **BM25 alone** → Misses semantic synonyms ("structuring" 6= "breaking deposits")
- **Dense alone** → Slow, prone to jargon mismatch
- **Hybrid** → Fast keyword matching + semantic understanding
- **RRF Fusion** → Combines both, unsupervised (no training needed)

### Why Reranking?
- Initial 30 results from BM25+Dense may have false positives
- CrossEncoder re-scores semantic relevance: query ↔ chunk
- Cuts to top-5 → LLM processes only highest-confidence chunks
- Improves: Answer quality, reduces hallucinations, faster generation

### Why Groq (not OpenAI)?
- **Cost**: Free tier available (vs. OpenAI pay-per-token)
- **Speed**: 70B LLM = better reasoning than GPT-3.5-turbo for AML
- **No censorship**: Groq < moderation flags for financial terminology
- **Multi-modal**: Groq Whisper transcription included

### Why Ollama (not cloud vision APIs)?
- **Privacy**: Graph captions generated locally, zero data leakage
- **Cost**: Free (vs. Google Vision API, Claude Vision)
- **Latency**: 7B LLaVA fast enough for 9 graphs
- **Fallback**: Mock captions work perfectly fine

### Why ChromaDB?
- **Lightweight**: No PostgreSQL/Qdrant infrastructure
- **Persistent**: Saves embeddings to disk
- **Python-native**: Simple `pip install`
- **Vector ops**: ANN search fast enough for <100K docs

---

## Future Enhancements

- [ ] **Fine-tuned reranker** on FFRAG Q&A pairs (supervised learning)
- [ ] **Real regulation PDFs**: Replace mocks with actual FATF/FinCEN documents
- [ ] **Time-series analysis**: Temporal pattern detection (velocity, acceleration)
- [ ] **Multi-user support**: Session management + audit logs
- [ ] **Webhook integration**: SAR filing auto-export to compliance tools
- [ ] **Graph anomaly detection**: Isolated subgraph clustering
- [ ] **Synthetic data scaling**: 10K+ transactions with controlled suspicion rates
- [ ] **Ensemble LLM**: Multiple models with weighted voting

---

## License & Citation

**Project**: FFRAG — Financial Forensics RAG Intelligence System  
**Author**: Financial Crime Analytics Team  
**Date**: March 2026  
**Status**: Production-ready

If using this system for research or publication, please cite:
```bibtex
@software{ffrag_2026,
  title={FFRAG: Financial Forensics Retrieval-Augmented Generation},
  author={You, Your Org},
  year={2026},
  url={https://github.com/yourorg/ffrag}
}
```

---

## Support & Contact

- **Issues**: Report bugs to [your issue tracker]
- **Questions**: Check the [Troubleshooting](#troubleshooting) section
- **Groq API**: https://console.groq.com
- **ChromaDB**: https://docs.trychroma.com
- **Ollama**: https://ollama.com

---

**Last Updated**: March 5, 2026  
**Version**: 2.0  
**Status**: ✅ Production-Ready
