# Q1: Multi-Agent RAG Orchestration

## The Agent Roles
1. **Retriever Agent:** Handles chunking (RecursiveCharacter) and vector store lookups (Chroma).
2. **Grader Agent (DeepAgents concept):** Evaluates retrieved documents against the query. If a document lacks semantic relevance, it is discarded. This is crucial for eliminating hallucinations in production code.
3. **Generator Agent:** Synthesizes the final answer strictly using the context verified by the Grader.

## Approach & Flow
We utilized LangGraph to orchestrate a state machine flow: `START -> Retrieve -> Grade -> Generate -> END`. 
- Groq's Llama-3 API is used for logic routing and generation. Rate limits are handled via max_retries implementation built into the LangChain ChatGroq wrapper.