from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import retrieve_verses, generate_response, is_relevant


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Bhagavad Gita RAG API",
    description="RAG-based Bhagavad Gita guidance assistant",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Request model
# ============================================================

class QueryRequest(BaseModel):
    query: str


# ============================================================
# Health check
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Bhagavad Gita RAG API is running"
    }


# ============================================================
# RAG endpoint
# ============================================================

@app.post("/ask")
def ask_question(request: QueryRequest):

    query = request.query.strip()

    if not query:
        return {
            "answer": "Please enter a question.",
            "verses": []
        }

    # Retrieve verses
    verses = retrieve_verses(query)

    # Relevance safeguard
    if not is_relevant(verses):
        return {
            "answer": (
                "I couldn't find sufficiently relevant Bhagavad Gita "
                "passages to answer this question reliably."
            ),
            "verses": []
        }

    # Generate grounded response
    answer = generate_response(query, verses)

    # Return useful verse information to frontend
    relevant_verses = []

    for verse in verses:
        relevant_verses.append({
            "verse_id": verse["verse_id"],
            "chapter": verse["chapter"],
            "verse": verse["verse"],
            "english": verse["english"],
            "similarity": round(verse["similarity"], 4)
        })

    return {
        "answer": answer,
        "verses": relevant_verses
    }