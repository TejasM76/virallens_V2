from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Optional, Dict
from .data_loader import setup_vector_retriever
from .agents import grade_documents, generate_answer

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    metadata_filter: Optional[Dict] # Added field for filtering

def retrieve(state):
    print("---RETRIEVING---")
    question = state["question"]
    
    # Safely get the filter if it exists in the state
    custom_filter = state.get("metadata_filter", None)
    
    # Pass the filter into our new hybrid setup
    retriever = setup_vector_retriever(metadata_filter=custom_filter) 
    documents = retriever.invoke(question)
    
    return {"documents": documents, "question": question, "metadata_filter": custom_filter}

def run_rag_pipeline(query: str):
    # If main.py is just doing the initial system load, build the DB and return early
    if query == "init":
        setup_vector_retriever()
        return "System Initialized."

    workflow = StateGraph(GraphState)
    
    # Define LangGraph Nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate_answer)
    
    # Build the Graph edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("grade_documents", "generate")
    workflow.add_edge("generate", END)
    
    app = workflow.compile()
    
    # Execute the graph
    result = app.invoke({"question": query})
    return result["generation"]