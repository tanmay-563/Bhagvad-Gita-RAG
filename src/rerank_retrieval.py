import pandas as pd
import numpy as np
import faiss
import pickle

from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# File paths
# ============================================================

EVALUATION_FILE = "data/evaluation_queries.csv"

INDEX_FILE = "data/gita_mpnet.index"
METADATA_FILE = "data/gita_mpnet_metadata.pkl"


# ============================================================
# Load evaluation dataset
# ============================================================

eval_df = pd.read_csv(EVALUATION_FILE)

print(f"Loaded {len(eval_df)} evaluation queries.")


# ============================================================
# Load MPNet embedding model
# ============================================================

print("\nLoading MPNet embedding model...")

embedding_model = SentenceTransformer(
    "all-mpnet-base-v2"
)


# ============================================================
# Load FAISS index
# ============================================================

index = faiss.read_index(INDEX_FILE)


# ============================================================
# Load metadata
# ============================================================

with open(METADATA_FILE, "rb") as f:
    metadata = pickle.load(f)


# ============================================================
# Load Cross-Encoder
# ============================================================

print("\nLoading Cross-Encoder reranker...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ============================================================
# Retrieval + Reranking
# ============================================================

def retrieve_and_rerank(query, candidate_k=20, final_k=10):

    # --------------------------------------------------------
    # Step 1: Encode query
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding)


    # --------------------------------------------------------
    # Step 2: Retrieve candidate verses using FAISS
    # --------------------------------------------------------

    scores, indices = index.search(
        query_embedding,
        candidate_k
    )


    # --------------------------------------------------------
    # Step 3: Build query-document pairs
    # --------------------------------------------------------

    pairs = []

    candidate_indices = indices[0]

    for idx in candidate_indices:

        document = metadata.iloc[idx]["document"]

        pairs.append(
            (query, document)
        )


    # --------------------------------------------------------
    # Step 4: Cross-Encoder scoring
    # --------------------------------------------------------

    rerank_scores = reranker.predict(
        pairs
    )


    # --------------------------------------------------------
    # Step 5: Sort by Cross-Encoder score
    # --------------------------------------------------------

    ranked_positions = np.argsort(
        rerank_scores
    )[::-1]


    # --------------------------------------------------------
    # Step 6: Return final ranked verses
    # --------------------------------------------------------

    results = []

    for position in ranked_positions[:final_k]:

        idx = candidate_indices[position]

        results.append({
            "verse_id": metadata.iloc[idx]["verse_id"],
            "chapter": metadata.iloc[idx]["chapter"],
            "verse": metadata.iloc[idx]["verse"],
            "document": metadata.iloc[idx]["document"],
            "score": float(rerank_scores[position])
        })


    return results


# ============================================================
# Evaluation
# ============================================================

recall_1 = []
recall_3 = []
recall_5 = []
recall_10 = []


print("\n" + "=" * 60)
print("MPNET + CROSS-ENCODER RERANKING")
print("=" * 60)


for _, row in eval_df.iterrows():

    query = row["query"]

    relevant_ids = {
        x.strip()
        for x in row["relevant_verse_ids"].split(";")
    }


    results = retrieve_and_rerank(
        query,
        candidate_k=20,
        final_k=10
    )


    retrieved_ids = [
        result["verse_id"]
        for result in results
    ]


    # --------------------------------------------------------
    # Calculate Recall@K
    # --------------------------------------------------------

    r1 = int(
        any(
            verse_id in relevant_ids
            for verse_id in retrieved_ids[:1]
        )
    )

    r3 = int(
        any(
            verse_id in relevant_ids
            for verse_id in retrieved_ids[:3]
        )
    )

    r5 = int(
        any(
            verse_id in relevant_ids
            for verse_id in retrieved_ids[:5]
        )
    )

    r10 = int(
        any(
            verse_id in relevant_ids
            for verse_id in retrieved_ids[:10]
        )
    )


    recall_1.append(r1)
    recall_3.append(r3)
    recall_5.append(r5)
    recall_10.append(r10)


    # --------------------------------------------------------
    # Print query result
    # --------------------------------------------------------

    print(f"\nQuery: {query}")

    print(f"Expected: {list(relevant_ids)}")

    print("Reranked:")

    for rank, result in enumerate(
        results[:5],
        start=1
    ):

        print(
            f"  {rank}. "
            f"{result['verse_id']} "
            f"(score={result['score']:.4f})"
        )


# ============================================================
# Overall metrics
# ============================================================

print("\n" + "=" * 60)
print("RERANKING EVALUATION RESULTS")
print("=" * 60)

print(
    f"Number of queries : {len(eval_df)}"
)

print(
    f"Recall@1          : {np.mean(recall_1):.3f}"
)

print(
    f"Recall@3          : {np.mean(recall_3):.3f}"
)

print(
    f"Recall@5          : {np.mean(recall_5):.3f}"
)

print(
    f"Recall@10         : {np.mean(recall_10):.3f}"
)

print("=" * 60)
