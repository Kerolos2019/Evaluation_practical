"""Tools the agent can call."""


from langchain_core.tools import tool

from src.vectorstore import get_collection



@tool
def search_knowledge_base(query: str, n_results: int = 4) -> str:
    """Search the internal knowledge base (CRM exports, projects, revenue, proposal
    documents) for information relevant to the query. Use this for any question
    about company data, accounts, opportunities, sales reps, projects, revenue,
    or proposals.
    """
    collection = get_collection()

    if collection.count() == 0:
        return "The knowledge base is empty."


    results = collection.query(query_texts=[query], n_results=n_results)

    documents = results.get("documents", [[]])[0]

    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return "No relevant information found in the knowledge base."

    formatted = []
   
    for text, meta in zip(documents, metadatas): 
        source = meta.get("source", "unknown") if meta else "unknown"
        formatted.append(f"[source: {source}]\n{text}")
  
  
    return "\n\n---\n\n".join(formatted)



@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location. This is a dummy
    implementation that returns a fixed response for demo purposes.
    """

    return f"The weather in {location} is sunny with a high of 24°C (dummy data)."



ALL_TOOLS = [search_knowledge_base, get_weather]
