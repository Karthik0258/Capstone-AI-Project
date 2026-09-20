import os
from typing import List, Literal
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.graph import StateGraph
from langgraph.types import TypedDict
from sentence_transformers import SentenceTransformer
import chromadb

class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

class State(TypedDict):
    query: str
    intent: Literal["policy_question", "general_question"]
    retrieved_chunks: List[dict]
    answer: str
    sources: List[str]
    confidence: float

model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("zepto_policies")

def classify_intent(state: State) -> State:
    q = state["query"].lower()
    keywords = ["delivery","return","refund","membership","tracking","cancel","gift card","support hours"]
    if any(k in q for k in keywords):
        state["intent"] = "policy_question"
    else:
        state["intent"] = "general_question"
    return state

def retrieve_and_answer(state: State) -> State:
    emb = model.encode(state["query"]).tolist()
    results = collection.query(query_embeddings=[emb], n_results=3)

    top_chunk = results["documents"][0][0][:200]  # first 200 chars
    sources = results["ids"][0]

    if os.getenv("MOCK_LLM", "1") == "1":
        answer = f"Based on the retrieved context: {top_chunk}"
    else:
        answer = f"(Real LLM would answer here using structured prompt)\nContext: {top_chunk}"

    state["retrieved_chunks"] = results["documents"][0]
    state["answer"] = answer
    state["sources"] = sources
    state["confidence"] = 1.0
    return state

def direct_answer(state: State) -> State:
    if os.getenv("MOCK_LLM", "1") == "1":
        answer = "I can only answer questions about Zepto policies right now."
    else:
        answer = "(Real LLM would answer here for general queries.)"

    state["answer"] = answer
    state["sources"] = []
    state["confidence"] = 1.0
    return state

graph = StateGraph(State)
graph.add_node("classify_intent", classify_intent)
graph.add_node("retrieve_and_answer", retrieve_and_answer)
graph.add_node("direct_answer", direct_answer)

graph.add_edge("classify_intent", "retrieve_and_answer", condition=lambda s: s["intent"]=="policy_question")
graph.add_edge("classify_intent", "direct_answer", condition=lambda s: s["intent"]=="general_question")

graph.set_entry_point("classify_intent")
graph.set_finish_point("retrieve_and_answer")
graph.set_finish_point("direct_answer")

app = FastAPI()

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    initial_state: State = {
        "query": req.query,
        "intent": None,
        "retrieved_chunks": [],
        "answer": "",
        "sources": [],
        "confidence": 0.0
    }
    final_state = graph.run(initial_state)
    return AskResponse(
        answer=final_state["answer"],
        sources=final_state["sources"],
        confidence=final_state["confidence"]
    )