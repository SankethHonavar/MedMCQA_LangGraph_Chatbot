import streamlit as st
from retriever import load_vector_store
from langgraph_graph import generate_answer
from time import sleep

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

    st.markdown("""
    <style>
    .fade-in {
        animation: fadeIn 0.7s ease-in;
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='fade-in'><h4>🧠 Answer:</h4></div>", unsafe_allow_html=True)
    answer_placeholder = st.empty()
    final_text = ""
    for char in response:
        final_text += char
        answer_placeholder.markdown(f"<div class='fade-in'>{final_text}</div>", unsafe_allow_html=True)
        sleep(0.01)

    with st.expander("🔎 Top Matches"):
        for i, doc in enumerate(results, 1):
            content = doc.page_content
            if query.lower() in content.lower():
                content = content.replace(query, f"**{query}**")
            st.markdown(f"**Result {i}:**\n\n{content}")

# 📬 Sidebar Contact
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📬 Contact")
    st.markdown("[📧 Email](mailto:sankethhonavar25@gmail.com)")
    st.markdown("[🔗 LinkedIn](https://linkedin.com/in/sankethhonavar)")
    st.markdown("[💻 GitHub](https://github.com/sankethhonavar)")

# ✨ Floating Icons (Right side - Top aligned)
st.markdown("""
<style>
.floating-button {
    position: fixed;
    top: 80px;
    right: 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    z-index: 9999;
}
.floating-button a {
    background-color: #0077b5;
    color: white;
    padding: 10px 14px;
    border-radius: 50%;
    text-align: center;
    font-size: 20px;
    text-decoration: none;
    box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.3);
    transition: background-color 0.3s;
}
.floating-button a:hover {
    background-color: #005983;
}
.floating-button a.email {
    background-color: #444444;
}
.floating-button a.email:hover {
    background-color: #222222;
}
.floating-button a.github {
    background-color: #171515;
}
.floating-button a.github:hover {
    background-color: #000000;
}
</style>

<div class="floating-button">
    <a href="mailto:sankethhonavar25@gmail.com" class="email" title="Email Me">
        <img src="https://img.icons8.com/ios-filled/25/ffffff/new-post.png" alt="Email"/>
    </a>
    <a href="https://linkedin.com/in/sankethhonavar" target="_blank" title="LinkedIn">
        <img src="https://img.icons8.com/ios-filled/25/ffffff/linkedin.png" alt="LinkedIn"/>
    </a>
    <a href="https://github.com/SankethHonavar" target="_blank" class="github" title="GitHub">
        <img src="https://img.icons8.com/ios-filled/25/ffffff/github.png" alt="GitHub"/>
    </a>
</div>
""", unsafe_allow_html=True)

# 📄 Footer
st.markdown("""
---
<p style='text-align: center; font-size: 0.9rem; color: grey'>
Made with ❤️ by <a href='https://linkedin.com/in/sankethhonavar' target='_blank'>Sanketh Honavar</a>
</p>
""", unsafe_allow_html=True)
