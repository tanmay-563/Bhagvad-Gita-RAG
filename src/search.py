import faiss
import pandas as pd

from sentence_transformers import SentenceTransformer


# ==========================================
# 1. Load FAISS index
# ==========================================

index = faiss.read_index("data/gita.index")


# ==========================================
# 2. Load metadata
# ==========================================

df = pd.read_pickle("data/gita_metadata.pkl")


# ==========================================
# 3. Load embedding model
# ==========================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# ==========================================
# 4. Search function
# ==========================================

def search_gita(query, top_k=5):

    # Convert query into embedding
    query_embedding = model.encode(
        [query]
    )

    # Convert to float32
    query_embedding = query_embedding.astype("float32")

    # Normalize
    faiss.normalize_L2(query_embedding)

    # Search FAISS
    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        result = {
            "verse_id": df.iloc[idx]["verse_id"],
            "chapter": df.iloc[idx]["chapter"],
            "verse": df.iloc[idx]["verse"],
            "score": float(score),
            "english": df.iloc[idx]["english"]
        }

        results.append(result)

    return results


# ==========================================
# 5. Test
# ==========================================

if __name__ == "__main__":

    query = input("\nEnter your problem/question: ")

    results = search_gita(query, top_k=5)

    print("\n" + "=" * 70)
    print("MOST RELEVANT GITA VERSES")
    print("=" * 70)

    for i, result in enumerate(results, start=1):

        print(f"\n{i}. {result['verse_id']}")
        print(
            f"Chapter {result['chapter']}, "
            f"Verse {result['verse']}"
        )

        print(
            f"Similarity: {result['score']:.4f}"
        )

        print("\n")
        print(result["english"])

        print("-" * 70)