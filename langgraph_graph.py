from dotenv import load_dotenv
load_dotenv()
from typing import TypedDict, List
from langgraph.graph import StateGraph
from retriever import load_vector_store
from langchain_google_genai import ChatGoogleGenerativeAI

from typing import Optional

class GraphState(TypedDict, total=False):
    query: str
    docs: List
    answer: Optional[str]
    proceed: Optional[bool]

llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    convert_system_message_to_human=True  # optional but recommended
)
db = load_vector_store()

def retrieve_node(state):
    query = state["query"]
    results = db.similarity_search(query, k=3)
    return {"query": query, "docs": results}

def filter_node(state):
    if not state["docs"]:
        return {"query": state["query"], "docs": [], "proceed": False}
    return {**state, "proceed": True}

def generate_node(state):
    context = "\n\n".join(doc.page_content for doc in state["docs"])
    prompt = (
        f"You are a helpful medical assistant. Use only the dataset context below to answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['query']}\n"
        f"If you are unsure, say 'Sorry, I couldn't find an answer based on the dataset.'"
    )
    
    print("📥 Prompt to LLM:\n", prompt)  # 👈 Debug prompt
    
    response = llm.invoke(prompt)
    
    print("📤 LLM Response:\n", response.content)  # 👈 Debug response
    
    return {"answer": response.content}

def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("filter", filter_node)
    builder.add_node("generate", generate_node)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "filter")
    builder.add_conditional_edges("filter", lambda x: "generate" if x["proceed"] else "end", {"generate", "end"})
    builder.set_finish_point("generate")

    return builder.compile()
