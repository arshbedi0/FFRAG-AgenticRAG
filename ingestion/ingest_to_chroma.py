"""
ingest_to_chroma.py
────────────────────
Ingests ALL three data sources into ChromaDB:
  1. Transaction CSV      → structured rows as text chunks
  2. Graph captions JSON  → image modality (LLaVA descriptions)
  3. Regulatory PDFs      → chunked regulation text

Run from project root:
  python ingestion/ingest_to_chroma.py

Collections created:
  - transactions   : individual transaction records
  - graph_captions : wallet network forensic descriptions
  - regulations    : FATF / FinCEN / OCC PDF chunks
"""

import os, json, re, csv
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG FROM .env ──
TRANSACTIONS_FILE = os.getenv("TRANSACTIONS_FILE", "data/transactions/saml_synthetic_1000.csv")
CAPTIONS_FILE     = os.getenv("CAPTIONS_FILE",     "data/graph_captions.json")
REGULATIONS_DIR   = os.getenv("REGULATIONS_DIR",   "data/regulations")
CHROMA_DIR        = os.getenv("CHROMA_DIR",         "chroma_db")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL",    "BAAI/bge-small-en-v1.5")

# Chunking config
PDF_CHUNK_SIZE    = int(os.getenv("PDF_CHUNK_SIZE",    "800"))   # characters
PDF_CHUNK_OVERLAP = int(os.getenv("PDF_CHUNK_OVERLAP", "150"))

# ── IMPORTS ──
try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    raise ImportError("Run: pip install chromadb sentence-transformers")


import pdfplumber


# ──────────────────────────────────────────────
# CHROMA SETUP
# ──────────────────────────────────────────────
print(f"\n{'='*60}")
print("  Financial Forensics RAG — Ingestion Pipeline")
print(f"{'='*60}\n")

print(f"📦 Initialising ChromaDB at: {CHROMA_DIR}")
client = chromadb.PersistentClient(path=CHROMA_DIR)

# Embedding function — runs locally, no API key needed
print(f"🔢 Loading embedding model: {EMBEDDING_MODEL}")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

# Create (or get) collections
col_transactions = client.get_or_create_collection(
    name="transactions",
    embedding_function=ef,
    metadata={"description": "SAML-D synthetic transaction records"}
)
col_captions = client.get_or_create_collection(
    name="graph_captions",
    embedding_function=ef,
    metadata={"description": "LLaVA wallet network forensic captions"}
)
col_regulations = client.get_or_create_collection(
    name="regulations",
    embedding_function=ef,
    metadata={"description": "FATF / FinCEN / OCC regulatory documents"}
)

print("✅ ChromaDB collections ready\n")


# ──────────────────────────────────────────────
# HELPER: BATCH UPSERT (Chroma has a 5461 limit)
# ──────────────────────────────────────────────
def batch_upsert(collection, ids, documents, metadatas, batch_size=500):
    total = len(ids)
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        collection.upsert(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end]
        )
    return total


# ──────────────────────────────────────────────
# SOURCE 1: TRANSACTION CSV
# ──────────────────────────────────────────────
print(f"{'─'*60}")
print("📊 SOURCE 1: Transaction Records (CSV)")
print(f"{'─'*60}")
print(f"   File: {TRANSACTIONS_FILE}")

def transaction_to_text(row: dict) -> str:
    """Convert a CSV row into a natural language sentence for embedding."""
    susp_label = "SUSPICIOUS" if str(row.get("Is_suspicious", "0")) == "1" else "normal"
    typology   = row.get("Type", "Unknown")
    amount     = float(row.get("Amount", 0))

    text = (
        f"Transaction on {row.get('Date', '')} at {row.get('Time', '')}: "
        f"Account {row.get('Sender_account', '')} in {row.get('Sender_bank_location', '')} "
        f"sent £{amount:,.2f} ({row.get('Payment_currency', '')}) "
        f"to account {row.get('Receiver_account', '')} in {row.get('Receiver_bank_location', '')} "
        f"received as {row.get('Received_currency', '')} "
        f"via {row.get('Payment_type', '')}. "
        f"Status: {susp_label}. "
        f"Typology: {typology}."
    )
    return text

ids, docs, metas = [], [], []

with open(TRANSACTIONS_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        doc_id = f"txn_{i:05d}"
        text   = transaction_to_text(row)
        meta   = {
            "source":                "transactions",
            "date":                  str(row.get("Date", "")),
            "sender_account":        str(row.get("Sender_account", "")),
            "receiver_account":      str(row.get("Receiver_account", "")),
            "amount":                float(row.get("Amount", 0)),
            "sender_location":       str(row.get("Sender_bank_location", "")),
            "receiver_location":     str(row.get("Receiver_bank_location", "")),
            "payment_type":          str(row.get("Payment_type", "")),
            "payment_currency":      str(row.get("Payment_currency", "")),
            "received_currency":     str(row.get("Received_currency", "")),
            "is_suspicious":         int(row.get("Is_suspicious", 0)),
            "typology":              str(row.get("Type", "Normal")),
        }
        ids.append(doc_id)
        docs.append(text)
        metas.append(meta)

total = batch_upsert(col_transactions, ids, docs, metas)
suspicious_count = sum(1 for m in metas if m["is_suspicious"] == 1)
print(f"   ✅ Ingested {total} transactions ({suspicious_count} suspicious, {total - suspicious_count} normal)\n")


# ──────────────────────────────────────────────
# SOURCE 2: GRAPH CAPTIONS (image modality)
# ──────────────────────────────────────────────
print(f"{'─'*60}")
print("🖼️  SOURCE 2: Graph Captions (Image Modality)")
print(f"{'─'*60}")
print(f"   File: {CAPTIONS_FILE}")

with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
    captions = json.load(f)

ids, docs, metas = [], [], []

for graph_id, cap in captions.items():
    caption_text = cap.get("caption", "")
    if not caption_text or len(caption_text.split()) < 10:
        print(f"   ⚠️  Skipping {graph_id} — caption too short")
        continue

    typology = cap.get("typology", "Unknown")
    countries = cap.get("countries", [])
    if isinstance(countries, list):
        countries_str = ", ".join(countries)
    else:
        countries_str = str(countries)

    # Enrich the caption with metadata context for better retrieval
    enriched = (
        f"[GRAPH ANALYSIS — {cap.get('title', typology)}]\n"
        f"Typology: {typology} | "
        f"Accounts: {cap.get('n_accounts', '?')} | "
        f"Transactions: {cap.get('n_transactions', '?')} | "
        f"Total Volume: £{cap.get('total_volume_gbp', 0):,} | "
        f"Suspicious Volume: £{cap.get('suspicious_volume', 0):,} | "
        f"Countries: {countries_str}\n\n"
        f"{caption_text}"
    )

    meta = {
        "source":           "graph_caption",
        "graph_id":         graph_id,
        "typology":         typology,
        "title":            cap.get("title", typology),
        "image_path":       cap.get("image_path", ""),
        "n_accounts":       int(cap.get("n_accounts") or 0),
        "n_transactions":   int(cap.get("n_transactions") or 0),
        "total_volume":     float(cap.get("total_volume_gbp") or 0),
        "suspicious_vol":   float(cap.get("suspicious_volume") or 0),
        "hub_account":      str(cap.get("hub_account", "")),
        "countries":        countries_str,
        "model":            cap.get("model", "unknown"),
    }

    ids.append(graph_id)
    docs.append(enriched)
    metas.append(meta)

total = batch_upsert(col_captions, ids, docs, metas)
print(f"   ✅ Ingested {total} graph captions\n")


# ──────────────────────────────────────────────
# SOURCE 3: REGULATORY PDFs
# ──────────────────────────────────────────────
print(f"{'─'*60}")
print("📄 SOURCE 3: Regulatory PDFs")
print(f"{'─'*60}")
print(f"   Directory: {REGULATIONS_DIR}\n")

def clean_text(text: str) -> str:
    """Clean extracted PDF text."""
    text = re.sub(r'\s+', ' ', text)           # collapse whitespace
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)  # fix hyphenation
    text = text.strip()
    return text

def semantic_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks, trying to break at sentence boundaries.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""

    for sentence in sentences:
        if len(current) + len(sentence) <= chunk_size:
            current += " " + sentence
        else:
            if current.strip():
                chunks.append(current.strip())
            # Start new chunk with overlap from previous
            if len(current) > overlap:
                current = current[-overlap:] + " " + sentence
            else:
                current = sentence

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if len(c) > 100]  # filter tiny fragments

reg_dir = Path(REGULATIONS_DIR)
pdf_files = list(reg_dir.glob("*.pdf"))

if not pdf_files:
    print("   ⚠️  No PDFs found in regulations directory")
    print(f"   Expected: {REGULATIONS_DIR}/*.pdf")
else:
    ids, docs, metas = [], [], []
    chunk_counter = 0

    for pdf_path in pdf_files:
        print(f"   📖 Processing: {pdf_path.name}")
        doc_name = pdf_path.stem

        try:
            full_text = ""
            with pdfplumber.open(pdf_path) as pdf:
                n_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        full_text += f"\n{page_text}"

            full_text = clean_text(full_text)

            if len(full_text) < 100:
                print(f"      ⚠️  Extracted text too short — PDF may be scanned/image-based")
                continue

            # Chunk the document
            chunks = semantic_chunk(full_text, PDF_CHUNK_SIZE, PDF_CHUNK_OVERLAP)
            print(f"      Pages: {n_pages} | Chars: {len(full_text):,} | Chunks: {len(chunks)}")

            for i, chunk in enumerate(chunks):
                chunk_id = f"reg_{doc_name}_{i:04d}"
                meta = {
                    "source":    "regulation",
                    "doc_name":  doc_name,
                    "filename":  pdf_path.name,
                    "chunk_idx": i,
                    "n_chunks":  len(chunks),
                }
                ids.append(chunk_id)
                docs.append(chunk)
                metas.append(meta)
                chunk_counter += 1

        except Exception as e:
            print(f"      ❌ Error processing {pdf_path.name}: {e}")

    if ids:
        total = batch_upsert(col_regulations, ids, docs, metas)
        print(f"\n   ✅ Ingested {total} regulation chunks from {len(pdf_files)} PDFs\n")


# ──────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────
print(f"{'='*60}")
print("  INGESTION COMPLETE")
print(f"{'='*60}")
print(f"  📊 transactions   : {col_transactions.count():>6,} documents")
print(f"  🖼️  graph_captions : {col_captions.count():>6,} documents")
print(f"  📄 regulations    : {col_regulations.count():>6,} documents")
print(f"  💾 ChromaDB path  : {CHROMA_DIR}/")
print(f"{'='*60}\n")
print("Next step: run  retrieval/retrieval_pipeline.py")


# ──────────────────────────────────────────────
# QUICK SANITY CHECK — test a sample query
# ──────────────────────────────────────────────
print("🔍 Sanity check — querying each collection...\n")

test_query = "suspicious structuring transaction below reporting threshold"

r1 = col_transactions.query(query_texts=[test_query], n_results=1)
print(f"  transactions → {r1['documents'][0][0][:120]}...")

r2 = col_captions.query(query_texts=[test_query], n_results=1)
print(f"  graph_captions → {r2['documents'][0][0][:120]}...")

r3 = col_regulations.query(query_texts=[test_query], n_results=1)
if r3['documents'][0]:
    print(f"  regulations → {r3['documents'][0][0][:120]}...")

print("\n✅ All collections responding. Ingestion verified.")
