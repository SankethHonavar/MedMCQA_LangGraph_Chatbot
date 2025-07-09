from retriever import retrieve_relevant_docs
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI

# LLM used for both doc chain and fallback answer
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)

# Define the structured prompt
prompt = PromptTemplate.from_template("""
You are a helpful medical assistant. Use only the dataset context below to answer.

Context:
{context}

Question: {input}

If you are unsure, say "Sorry, I couldn't find an answer based on the dataset." Do not guess.
""")

# Build document chain and retrieval chain
document_chain = create_stuff_documents_chain(llm, prompt)
retriever_chain = create_retrieval_chain(retrieve_relevant_docs(), document_chain)

# Expose chain for Streamlit app
graph = retriever_chain

# Manual fallback function if needed
def generate_answer(query: str, context: str) -> str:
    if not context.strip():
        return "Sorry, I couldn't find an answer based on the dataset."

    fallback_llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)
    fallback_prompt = f"""
    You are a helpful medical assistant. Use only the dataset context below to answer.

    Context:
    {context}

    Question: {query}

    If you are unsure, say "Sorry, I couldn't find an answer based on the dataset." Do not guess.
    """
    response = fallback_llm.invoke(fallback_prompt)
    return response.content.strip()
