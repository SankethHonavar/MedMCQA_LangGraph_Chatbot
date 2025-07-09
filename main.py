import os
from retriever import load_vector_store
from langgraph_graph import generate_answer


def medchat(query):
    """
    Full MedMCQA pipeline.
    1. Retrieve top matches
    2. Prompt LLM with strict instruction to avoid hallucination
    """
    retriever = load_vector_store()
    matches = retriever.similarity_search(query, k=3)
    context = "\n\n".join([match.page_content for match in matches])

    prompt = f"""
    You are a helpful medical assistant. Use only the dataset context below to answer.

    Context:
    {context}

    Question: {query}
    If you are unsure, say 'Sorry, I couldn't find an answer based on the dataset.'
    """

    return generate_answer(prompt)


if __name__ == "__main__":
    print("\n🩺 MedMCQA Chatbot")
    print("Ask a medical question and get answers from MedMCQA dataset.\n")

    while True:
        user_q = input("Ask a medical question (or type 'exit'): ")
        if user_q.lower() == "exit":
            break
        response = medchat(user_q)
        print("\n🧠 Answer:", response, "\n")
