from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

# We use Groq with Llama 3 for fast, free inference. 
# Rate limiting is handled natively by Groq API SDK retries, but we keep requests isolated.
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_retries=3)

def grade_documents(state):
    """Grader Agent: Filters out irrelevant documents to prevent hallucinations."""
    print("---GRADING DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]
    
    prompt = ChatPromptTemplate.from_template(
        "You are a grader assessing relevance of a document to a user question.\n"
        "Here is the document:\n{document}\n"
        "Here is the question: {question}\n"
        "If the document contains keywords or semantic meaning related to the question, grade it as 'yes'. Otherwise, 'no'. Just answer 'yes' or 'no'."
    )
    
    chain = prompt | llm | StrOutputParser()
    filtered_docs = []
    
    for d in documents:
        score = chain.invoke({"question": question, "document": d.page_content})
        if "yes" in score.lower():
            filtered_docs.append(d)
            
    return {"documents": filtered_docs, "question": question}

def generate_answer(state):
    """Generator Agent: Synthesizes the final answer using only graded documents."""
    print("---GENERATING ANSWER---")
    question = state["question"]
    documents = state["documents"]
    
    context = "\n\n".join(doc.page_content for doc in documents)
    
    prompt = ChatPromptTemplate.from_template(
        "Use the following pieces of retrieved context to answer the question.\n"
        "If you don't know the answer, say that you don't know.\n"
        "Question: {question}\nContext: {context}\nAnswer:"
    )
    
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({"context": context, "question": question})
    
    return {"documents": documents, "question": question, "generation": generation}