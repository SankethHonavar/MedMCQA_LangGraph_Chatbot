import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os

# Set up FAISS and embeddings
@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("data/medmcqa_index", embeddings, allow_dangerous_deserialization=True)

# Page config
st.set_page_config(page_title="MedMCQA Chatbot", layout="centered")
st.title("🩺 MedMCQA Chatbot")
st.markdown("Ask a medical question and get answers from MedMCQA dataset.")

# Load DB
db = load_vector_store()

# User input
query = st.text_input("🔍 Enter your medical question:")

# Search
if query:
    with st.spinner("Searching..."):
        results = db.similarity_search(query, k=3)

    st.markdown("### 🔎 Top Matches:")
    for i, doc in enumerate(results, 1):
        st.markdown(f"**Result {i}:**")
        st.code(doc.page_content, language="text")
        st.markdown("---")

# Footer
st.caption("🔬 Powered by LangChain + HuggingFace + FAISS + MedMCQA")
