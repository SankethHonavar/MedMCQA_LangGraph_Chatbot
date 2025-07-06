from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dataset_loader import load_medmcqa_subset
from tqdm import tqdm  # Progress bar for better visibility during indexing
import os

os.environ["GOOGLE_API_KEY"] = "GOOGLE_API_KEY"

def create_vector_store():
    examples = load_medmcqa_subset()

    # Convert to LangChain documents with progress bar
    docs = [
        Document(
            page_content=e["formatted"],
            metadata={"question": e["question"]}
        )
        for e in tqdm(examples, desc="📚 Embedding MedMCQA examples")
    ]

    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build and save FAISS index
    db = FAISS.from_documents(docs, embeddings)
    os.makedirs("data", exist_ok=True)
    db.save_local("data/medmcqa_index")
    print("✅ Vector DB saved at data/medmcqa_index")

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("data/medmcqa_index", embeddings, allow_dangerous_deserialization=True)

# 🚀 Test block
if __name__ == "__main__":
    if not os.path.exists("data/medmcqa_index"):
        create_vector_store()

    db = load_vector_store()
    query = "What is the treatment for asthma?"
    results = db.similarity_search(query, k=3)  # Get top 3 results

    print("\n🔍 Top Similar Documents:\n")
    for i, doc in enumerate(results, 1):
        print(f"🔹 Result {i}:\n{doc.page_content}\n{'-'*60}")