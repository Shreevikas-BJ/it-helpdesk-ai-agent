# app/retriever.py
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from .config import KB_INDEX_DIR

def load_kb():
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(KB_INDEX_DIR, emb, allow_dangerous_deserialization=True)
