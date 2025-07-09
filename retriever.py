from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dataset_loader import load_medmcqa_subset
from tqdm import tqdm  # Progress bar for better visibility during indexing
import os

def retrieve_relevant_docs():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("data/medmcqa_index", embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

def create_vector_store():
    examples = load_medmcqa_subset()

    # Format each entry into a LangChain Document with progress bar
    docs = [
        Document(
            page_content=e["formatted"],
            metadata={"question": e["question"]}
        )
        for e in tqdm(examples, desc="📚 Embedding MedMCQA examples")
    ]

    # Create embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build and save FAISS index
    db = FAISS.from_documents(docs, embeddings)
    os.makedirs("data", exist_ok=True)
    db.save_local("data/medmcqa_index")
    print("✅ Vector DB saved at data/medmcqa_index")


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("data/medmcqa_index", embeddings, allow_dangerous_deserialization=True)


if __name__ == "__main__":
    from langchain.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_google_genai import ChatGoogleGenerativeAI

    # Load DB
    db = load_vector_store()
    query = "What is the treatment for asthma?"
    docs = db.similarity_search(query, k=4)

    # Prompt Template
    prompt_template = PromptTemplate.from_template(
        """
        You are a helpful medical assistant. Use only the dataset context below to answer.

        Context:
        {context}

        Question:
        {question}

        If you are unsure, say 'Sorry, I couldn't find an answer based on the dataset.'
        """
    )

    # LLM
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)

    chain = prompt_template | llm | StrOutputParser()
    print("\n\n🧠 Answer:\n", chain.invoke({"context": "\n\n".join(d.page_content for d in docs), "question": query}))
