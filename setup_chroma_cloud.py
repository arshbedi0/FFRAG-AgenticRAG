"""
setup_chroma_cloud.py
─────────────────────
Initialize ChromaDB on Streamlit Cloud by downloading and unzipping a pre-built database.
This runs once at startup to set up the vector embeddings.
"""

import os
import sys
import shutil
import zipfile
import urllib.request
import streamlit as st
from pathlib import Path

CHROMA_DIR = "chroma_db"
CHROMA_ZIP_URL = "https://github.com/arshbedi0/FFRAG-AgenticRAG/releases/download/v1.0/chroma_db.zip"
CHROMA_ZIP_FILE = "chroma_db_temp.zip"


def setup_chroma_db():
    """Download and extract ChromaDB if not already present."""
    
    # Check if ChromaDB already exists
    if os.path.exists(CHROMA_DIR) and os.path.isfile(f"{CHROMA_DIR}/chroma.sqlite3"):
        print(f"✅ ChromaDB already initialized at {CHROMA_DIR}")
        return True
    
    print(f"📥 ChromaDB not found. Downloading from GitHub...")
    
    try:
        # Download the ZIP file
        print(f"   Downloading {CHROMA_ZIP_URL}...")
        urllib.request.urlretrieve(CHROMA_ZIP_URL, CHROMA_ZIP_FILE)
        print(f"   ✅ Downloaded {CHROMA_ZIP_FILE}")
        
        # Extract to chroma_db directory
        print(f"   Extracting to {CHROMA_DIR}...")
        with zipfile.ZipFile(CHROMA_ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(CHROMA_DIR)
        print(f"   ✅ Extracted successfully")
        
        # Clean up ZIP file
        if os.path.exists(CHROMA_ZIP_FILE):
            os.remove(CHROMA_ZIP_FILE)
            print(f"   🧹 Cleaned up temporary ZIP file")
        
        # Verify extraction
        if os.path.isfile(f"{CHROMA_DIR}/chroma.sqlite3"):
            print(f"✅ ChromaDB setup complete! Database ready.")
            return True
        else:
            print(f"❌ ChromaDB extraction failed: chroma.sqlite3 not found")
            return False
            
    except Exception as e:
        print(f"❌ Error setting up ChromaDB: {str(e)}")
        print(f"   Please check the URL and try again.")
        return False


def get_secrets():
    """Get secrets from Streamlit secrets manager."""
    try:
        secrets = {
            "groq_api_key": st.secrets.get("groq_api_key", ""),
            "neo4j_uri": st.secrets.get("neo4j_uri", ""),
            "neo4j_user": st.secrets.get("neo4j_user", "neo4j"),
            "neo4j_password": st.secrets.get("neo4j_password", ""),
            "huggingface_token": st.secrets.get("huggingface_token", ""),
        }
        return secrets
    except Exception as e:
        print(f"⚠️  Could not load secrets from Streamlit: {str(e)}")
        print(f"   Make sure to set secrets in Streamlit Cloud dashboard")
        return None


if __name__ == "__main__":
    setup_chroma_db()
