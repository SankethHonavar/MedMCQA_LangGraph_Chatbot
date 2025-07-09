# 🩺 MedMCQA LangGraph Chatbot

A **context-aware medical Q&A chatbot** powered by **LangGraph**, **Google Gemini**, and the **MedMCQA dataset**.  
It only answers **factually** using dataset evidence, and **gracefully declines** if no relevant answer exists—**no hallucinations!**

![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-orange?style=for-the-badge)
![LangGraph](https://img.shields.io/badge/Uses-LangGraph-blue?style=for-the-badge)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-green?style=for-the-badge)
![Google Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-red?style=for-the-badge)

---

## 📸 Demo
https://medmcapplanggraphchatbot-v7xvxen5njjmtonccse4g7.streamlit.app/

---

## 🚀 Features

- ✅ Query strictly answers from **MedMCQA dataset**
- ❌ No hallucinations — responds **only when confident**
- 💬 Uses **LangGraph** for flow control
- 🧠 Powered by **Google Gemini** via `langchain-google-genai`
- 🔍 Efficient similarity search with **FAISS** + **HuggingFace embeddings**
- 🧾 Built with an intuitive **Streamlit UI**

---

## 📦 Installation

```bash
git clone https://github.com/SankethHonavar/MedMCQA_LangGraph_Chatbot.git
cd MedMCQA_LangGraph_Chatbot
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
````

---

## 🔐 Setup API Key

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

---

## 🧠 Generate Vector Store (One-time)

```bash
python main.py
```

This loads 5,000 QA pairs from MedMCQA and stores embeddings in `data/medmcqa_index`.

---

## 💬 Run the Chatbot

```bash
streamlit run app.py
```

Then open the Streamlit interface in your browser and start asking questions like:

* “What is the treatment for asthma?”
* “Which drug is contraindicated in pregnancy?”

If a question is not in the dataset, you'll get:

> ❌ "Sorry, I couldn't find an answer based on the dataset."

---

## 📁 Project Structure

```
├── app.py                  # Streamlit frontend
├── main.py                 # Builds vector store from MedMCQA
├── dataset_loader.py       # Loads MedMCQA dataset
├── retriever.py            # Embedding + FAISS retriever
├── langgraph_graph.py      # LangGraph flow logic
├── requirements.txt
└── data/
    └── medmcqa_index/      # Vector DB storage
```

---

## 🧪 Dataset

We use the [MedMCQA](https://huggingface.co/datasets/medmcqa) dataset from HuggingFace.
It's a large-scale, multiple-choice QA dataset curated from medical entrance exams like NEET-PG.

---

## 🤝 Credits

* 🧠 Dataset: [MedMCQA](https://huggingface.co/datasets/medmcqa)
* 🤖 LLM: [Gemini 1.5 Flash](https://aistudio.google.com/)
* 🔗 LangGraph: [LangGraph Framework](https://www.langgraph.dev/)
* 💬 Chat UI: [Streamlit](https://streamlit.io)

---

## ✉️ Contact

[📧 sankethhonavar25@gmail.com](mailto:sankethhonavar25@gmail.com)
[🔗 LinkedIn](https://linkedin.com/in/sankethhonavar)

---

## ⭐ Star this Repo

If you found this helpful, consider starring 🌟 the repo to support more work like this!

```

---

Would you like me to:
- Generate this file and add it to your repo as `README.md`?
- Help record a quick 1-min screen demo video?

Let me know!
```
