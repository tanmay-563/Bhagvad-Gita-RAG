import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from google import genai
from pathlib import Path

# ============================================================
# Configuration
# ============================================================


BASE_DIR = Path(__file__).resolve().parent.parent

INDEX_FILE = str(BASE_DIR / "data" / "gita_mpnet.index")
METADATA_FILE = str(BASE_DIR / "data" / "gita_mpnet_metadata.pkl")
TOP_K = 5

MODEL_NAME = "gemini-3.8-flash"


# ============================================================
# Initialize Gemini
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY environment variable not found."
    )

client = genai.Client(api_key=api_key)


# ============================================================
# Load retrieval components
# ============================================================

print("Loading MPNet embedding model...")

embedding_model = SentenceTransformer(
    "all-mpnet-base-v2"
)

print("Loading FAISS index...")

index = faiss.read_index(INDEX_FILE)


print("Loading verse metadata...")

with open(METADATA_FILE, "rb") as f:
    metadata = pickle.load(f)


# ============================================================
# Retrieve relevant verses
# ============================================================

def retrieve_verses(query, top_k=TOP_K):

    # Create query embedding
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding)

    # Search FAISS
    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        row = metadata.iloc[idx]

        results.append({
            "verse_id": row["verse_id"],
            "chapter": int(row["chapter"]),
            "verse": int(row["verse"]),
            "english": row["english"],
            "sanskrit": row["sanskrit"],
            "transliteration": row["transliteration"],
            "similarity": float(score)
        })

    return results
# ============================================================
# Relevance safeguard
# ============================================================

RELEVANCE_THRESHOLD = 0.25


def is_relevant(verses):
    """
    Check whether the retrieved passages have enough
    semantic similarity to the user's query.
    """

    top_score = verses[0]["similarity"]

    return top_score >= RELEVANCE_THRESHOLD

# ============================================================
# Generate grounded response
# ============================================================

def generate_response(query, verses):

    context_parts = []

    for verse in verses:

        context_parts.append(
            f"""
Chapter {verse['chapter']}, Verse {verse['verse']}
Verse ID: {verse['verse_id']}

English:
{verse['english']}
"""
        )

    context = "\n".join(context_parts)


    prompt = f"""
You are a Bhagavad Gita guidance assistant.

The user has described a problem or asked a question.

Answer the user's question using ONLY the Bhagavad Gita passages
provided in the context below.

Important rules:

1. Do not invent Bhagavad Gita verses.
2. Do not claim that a verse says something that is not supported
   by the provided text.
3. Explain the guidance in simple, practical language.
4. Mention the relevant chapter and verse numbers when using them.
5. If the retrieved passages do not directly answer the question,
   clearly say that the retrieved passages do not directly address it.
6. Do not pretend to provide professional medical, legal, or
   financial advice.
7. Keep the response concise but meaningful.

User question:
{query}

Retrieved Bhagavad Gita passages:
{context}

Provide:

1. A direct explanation.
2. The relevant verse references.
3. A short practical takeaway.
"""


    interaction = client.interactions.create(
    model=MODEL_NAME,
    input=prompt,
    generation_config={
        "thinking_level": "low"
     }
    )
    return interaction.output_text
    





# ============================================================
# Main application
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("BHAGAVAD GITA RAG ASSISTANT")
    print("=" * 60)

    query = input("\nDescribe your problem or ask a question:\n> ")

    print("\nSearching relevant verses...\n")

    verses = retrieve_verses(query)

    print("=" * 60)
    print("RETRIEVED VERSES")
    print("=" * 60)

    for rank, verse in enumerate(verses, start=1):

        print(
            f"\n{rank}. "
            f"{verse['verse_id']} "
            f"(similarity={verse['similarity']:.4f})"
        )

        print(verse["english"])


    
    print("\n" + "=" * 60)
    print("GENERATING RESPONSE...")
    print("=" * 60)

    if not is_relevant(verses):
        answer = (
            "I couldn't find sufficiently relevant Bhagavad Gita passages "
            "to answer this question reliably. Please ask a question related "
            "to the themes or teachings covered in the Bhagavad Gita."
        )
    else:
        answer = generate_response(
            query,
            verses
        )
        

    print("\n" + "=" * 60)
    print("GITA GUIDANCE")
    print("=" * 60)

    print(answer)


if __name__ == "__main__":
    main()