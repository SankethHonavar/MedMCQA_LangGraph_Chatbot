from langgraph_graph import build_graph

def medchat(query):
    graph = build_graph()
    result = graph.invoke({"query": query})
    return result.get("answer", "Sorry, I couldn't find an answer.")

if __name__ == "__main__":
    while True:
        user_q = input("\nAsk a medical question (or type 'exit'): ")
        if user_q.lower() == "exit":
            break
        response = medchat(user_q)
        print(f"\n🧠 Answer:\n{response}")
