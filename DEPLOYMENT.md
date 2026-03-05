# FFRAG Deployment Guide — Streamlit Cloud

## 🚀 Quick Start (5 minutes)

### Step 1: Prepare ChromaDB for Upload

First, zip your local ChromaDB and upload to GitHub Releases:

```bash
# On your local machine
cd d:\FFRAGCB

# Zip the ChromaDB directory
tar -czf chroma_db.tar.gz chroma_db/
# OR on Windows:
# (Use 7-Zip or WinRAR to create chroma_db.zip)
```

Then upload `chroma_db.zip` to your GitHub repository as a Release:
1. Go to https://github.com/arshbedi0/FFRAG-AgenticRAG/releases
2. Click "Create a new release"
3. Tag: `v1.0`
4. Upload `chroma_db.zip` as an asset
5. Copy the download URL (should be like: `https://github.com/arshbedi0/FFRAG-AgenticRAG/releases/download/v1.0/chroma_db.zip`)

### Step 2: Connect GitHub to Streamlit Cloud

1. Go to https://streamlit.io/cloud
2. Click "New app"
3. Select your GitHub repo: `arshbedi0/FFRAG-AgenticRAG`
4. Select branch: `main`
5. Set main file path: `ui/app.py`
6. Click "Deploy"

### Step 3: Add Secrets to Streamlit Cloud

1. Go to your app's settings (gear icon)
2. Click "Secrets"
3. Paste this into the text editor:

```toml
# API Keys
groq_api_key = "gsk_YOUR_ACTUAL_KEY_HERE"

# Neo4j AuraDB
neo4j_uri = "neo4j+s://xxxx.databases.neo4j.io"
neo4j_user = "neo4j"
neo4j_password = "your-password"

# ChromaDB will be auto-downloaded on first run
```

4. Click "Save"

### Step 4: Verify Deployment

1. Your app will deploy automatically
2. First load will download & unzip ChromaDB (~5 min)
3. Subsequent loads use cached ChromaDB (fast)

**✅ Your app is live!**

---

## 📦 What Gets Deployed

```
Streamlit Cloud Container:
├── ui/app.py                    (main UI)
├── retrieval/                   (search pipeline)
├── generation/                  (Groq LLM)
├── ingestion/                   (data processing)
├── evaluation/                  (metrics)
├── requirements.txt             (dependencies)
├── setup_chroma_cloud.py        (auto-init ChromaDB)
└── .streamlit/config.toml       (theme settings)

External Services:
├── ChromaDB (auto-downloaded as ZIP on first run)
├── Neo4j AuraDB (cloud-hosted graph DB)
└── Groq API (LLM generation)
```

---

## 🔧 Configuration Details

### Required Environment Variables (set in Streamlit Secrets):

| Variable | Source | Purpose |
|----------|--------|---------|
| `groq_api_key` | https://console.groq.com/keys | LLM API key |
| `neo4j_uri` | Neo4j AuraDB console | Graph database connection |
| `neo4j_user` | `neo4j` (default) | Graph DB username |
| `neo4j_password` | Neo4j AuraDB console | Graph DB password |
| `huggingface_token` | https://huggingface.co/settings/tokens | Optional for private models |

### How ChromaDB Works on Streamlit Cloud

1. **First Visit**: 
   - App starts
   - `setup_chroma_cloud.py` runs automatically
   - Downloads `chroma_db.zip` from GitHub (~2min, ~150MB)
   - Unzips to `/chroma_db/` 
   - Initializes collections
   - Ready for queries ✅

2. **Subsequent Visits**:
   - ChromaDB already extracted
   - Loads from disk instantly
   - Queries execute normally

3. **Persistent Storage**:
   - Streamlit Cloud doesn't persist files by default
   - **Solution**: Re-download on each app restart (costs ~5s, minimal)
   - OR upgrade to Streamlit Cloud Pro for persistent storage

---

## 🔐 Security Best Practices

✅ **Already configured:**
- All secrets in Streamlit Secrets Manager (not in code)
- `.env` excluded from git
- API keys never logged
- HTTPS only

❌ **Never do this:**
- Commit `.env` to GitHub
- Hardcode API keys in Python files
- Use `gsk_` tokens in public repos

---

## 📊 Performance Expectations

| Operation | Time | Notes |
|-----------|------|-------|
| App cold start | 30-60s | First deployment, Streamlit boots |
| App warm start | 5-10s | Subsequent reloads |
| ChromaDB init (1st time) | 120-300s | Downloads + unzips |
| ChromaDB load (cached) | <1s | Already extracted |
| Query execution | 2-5s | Retrieval + LLM generation |
| Neo4j graph query | 1-2s | Multi-hop traversal |

**Tip**: For faster reloads, keep app running and use "Always rerun" in dev mode.

---

## 🐛 Troubleshooting

### "ChromaDB initialization failed"
```
❌ Error: Could not download chroma_db.zip
✅ Fix: 
  1. Check GitHub Release URL is correct
  2. Verify ZIP file is accessible
  3. Try uploading a fresh ZIP
```

### "Neo4j connection refused"
```
❌ Error: neo4j+s://... connection failed
✅ Fix:
  1. Verify Neo4j AuraDB instance is running
  2. Check credentials in Streamlit Secrets
  3. Ensure IP whitelist allows Streamlit Cloud IPs
```

### "Secrets not loading"
```
❌ Error: KeyError: 'groq_api_key'
✅ Fix:
  1. Go to app settings → Secrets
  2. Verify all required variables are set
  3. Use exact key names (case-sensitive)
  4. Restart app after changing secrets
```

### "Out of memory"
```
❌ Error: MemoryError during query
✅ Fix:
  1. Reduce TOP_K_EACH in query (from 10 to 5)
  2. Increase Streamlit Cloud plan
  3. Use streaming for long responses
```

---

## 📈 Scaling After Launch

### If you need better performance:

**Option A: Upgrade Streamlit Plan**
- Standard: $5/month (1 GB RAM) ← Current
- Pro: $30/month (4 GB RAM) + persistent storage
- Business: Custom pricing

**Option B: Move to Container (Azure/AWS)**
- More control over resources
- Can use GPUs for faster inference
- More expensive (~$10-50/month)

**Option C: Optimize Code**
- Use streaming for large responses
- Cache embedding model in memory
- Reduce TOP_K parameters
- Use async calls

---

## 🎯 Post-Deployment Checklist

- [ ] App loads without errors
- [ ] ChromaDB initializes on first visit
- [ ] Neo4j queries work
- [ ] Groq LLM generation works
- [ ] Voice input transcription works
- [ ] Guardrails functioning
- [ ] Response formatting correct
- [ ] Source citations appear

---

## 📞 Support & Resources

- **Streamlit Docs**: https://docs.streamlit.io/
- **Streamlit Cloud Docs**: https://docs.streamlit.io/streamlit-cloud/
- **Groq Console**: https://console.groq.com/
- **Neo4j Aura**: https://console.neo4j.io/
- **GitHub Releases API**: https://docs.github.com/rest/releases/

---

## 🔄 Updating After Deployment

### Update Python Code:
1. Commit to GitHub
2. Streamlit Cloud auto-redeploys
3. App updates within 1 minute

### Update ChromaDB Embeddings:
1. Delete local `chroma_db/`
2. Run re-ingestion: `python ingestion/ingest_to_chroma.py`
3. Zip new `chroma_db_updated.zip`
4. Upload to GitHub Release
5. Update URL in `setup_chroma_cloud.py`
6. Redeploy: Streamlit Cloud will auto-download

### Update Secrets:
1. Go to app settings
2. Edit Secrets
3. Changes apply immediately on next run

---

**Deployed! 🎉** Your FFRAG system is now live at your Streamlit Cloud URL.
