import streamlit as st
from retriever import load_vector_store
from langgraph_graph import generate_answer

# Load vector DB
db = load_vector_store()

st.set_page_config("MedMCQA Chatbot", page_icon="🩺")

# 🌗 Theme toggle sidebar
with st.sidebar:
    st.title("🩺 MedMCQA Chatbot")
    theme_mode = st.radio("🌓 Theme", ["Light", "Dark"], horizontal=True)

# 🌓 Apply selected theme
if theme_mode == "Dark":
    st.markdown("""
        <style>
        :root { --text-color: #eee; }
        body, .stApp {
            background-color: #1e1e1e !important;
            color: var(--text-color) !important;
        }
        .stTextInput input {
            background-color: #333 !important;
            color: var(--text-color) !important;
        }
        .stTextInput label {
            color: var(--text-color) !important;
        }
        input::placeholder {
            color: #bbb !important;
        }
        .stButton>button {
            background-color: #444 !important;
            color: var(--text-color) !important;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        :root { --text-color: #111; }
        body, .stApp {
            background-color: #ffffff !important;
            color: var(--text-color) !important;
        }
        .stTextInput input {
            background-color: #f0f0f0 !important;
            color: var(--text-color) !important;
        }
        .stTextInput label {
            color: var(--text-color) !important;
        }
        input::placeholder {
            color: #444 !important;
        }
        .stButton>button {
            background-color: #e0e0e0 !important;
            color: var(--text-color) !important;
        }
        </style>
    """, unsafe_allow_html=True)

# 🧠 App title
st.header("🩺 MedMCQA Chatbot")
st.caption("Ask a medical question and get answers from the MedMCQA dataset only. If not found, it will respond gracefully.")

# ✏️ Query box
query = st.text_input(
    "🔍 Enter your medical question:",
    placeholder="e.g., What is the mechanism of Aspirin?",
    label_visibility="visible"
)

# 🚀 Answer generation
if query:
    results = db.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in results])

    with st.spinner("🧠 Generating answer..."):
        response = generate_answer(query, context)

    st.markdown("### 🔎 Top Matches:")
    for i, doc in enumerate(results, 1):
        st.markdown(f"**Result {i}:**\n\n{doc.page_content}")

    st.markdown("---")
    st.markdown("### 🧠 Answer:")
    st.success(response)

# 📬 Sidebar Contact
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📬 Contact")
    st.markdown("[📧 Email](mailto:sankethhonavar25@gmail.com)")
    st.markdown("[🔗 LinkedIn](https://linkedin.com/in/sankethhonavar)")
    st.markdown("[💻 GitHub](https://github.com/sankethhonavar)")

# ✨ Floating Icons (Right side - Updated)
st.markdown("""
<style>
.floating-icons {
    position: fixed;
    top: 80px;
    right: 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    z-index: 9999;
}
.floating-icons a {
    width: 42px;
    height: 42px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: #0077b5;
    color: white;
    text-decoration: none;
    font-size: 22px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.floating-icons a.email {
    background-color: #d44638;
}
.floating-icons a.github {
    background-color: #000000;
}
</style>

<div class="floating-icons">
    <a href="mailto:sankethhonavar25@gmail.com" class="email" title="Email">📩</a>
    <a href="https://linkedin.com/in/sankethhonavar" title="LinkedIn">💼</a>
    <a href="https://github.com/sankethhonavar" class="github" title="GitHub">🐙</a>
</div>
""", unsafe_allow_html=True)
