import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer


# -----------------------------
# File paths
# -----------------------------
EVALUATION_FILE = "data/evaluation_queries.csv"
INDEX_FILE = "data/gita_mpnet.index"
METADATA_FILE = "data/gita_mpnet_metadata.pkl"


# -----------------------------
# Load evaluation dataset
# -----------------------------
df = pd.read_csv(EVALUATION_FILE)

print(f"Loaded {len(df)} evaluation queries.")


# -----------------------------
# Load FAISS index
# -----------------------------
index = faiss.read_index(INDEX_FILE)


# -----------------------------
# Load metadata
# -----------------------------
with open(METADATA_FILE, "rb") as f:
    metadata = pickle.load(f)


# -----------------------------
# Load embedding model
# -----------------------------
model = SentenceTransformer("all-mpnet-base-v2")


# -----------------------------
# Evaluation function
# -----------------------------
def evaluate_query(query, relevant_ids, top_k=10):

    # Convert relevant IDs into a set
    relevant_ids = set(relevant_ids)

    # Create query embedding
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding)

    # Search
    scores, indices = index.search(query_embedding, top_k)

    retrieved_ids = []

    for idx in indices[0]:
     retrieved_ids.append(metadata.iloc[idx]["verse_id"])

    # Calculate recall at different K values
    recall_1 = int(any(
        verse_id in relevant_ids
        for verse_id in retrieved_ids[:1]
    ))

    recall_3 = int(any(
        verse_id in relevant_ids
        for verse_id in retrieved_ids[:3]
    ))

    recall_5 = int(any(
        verse_id in relevant_ids
        for verse_id in retrieved_ids[:5]
    ))

    recall_10 = int(any(
        verse_id in relevant_ids
        for verse_id in retrieved_ids[:10]
    ))

    return {
        "query": query,
        "relevant": list(relevant_ids),
        "retrieved": retrieved_ids,
        "recall@1": recall_1,
        "recall@3": recall_3,
        "recall@5": recall_5,
        "recall@10": recall_10
    }


# -----------------------------
# Run evaluation
# -----------------------------
results = []

for _, row in df.iterrows():

    query = row["query"]

    relevant_ids = [
        x.strip()
        for x in row["relevant_verse_ids"].split(";")
    ]

    result = evaluate_query(
        query,
        relevant_ids,
        top_k=10
    )

    results.append(result)


# -----------------------------
# Calculate overall metrics
# -----------------------------
recall_at_1 = np.mean([
    result["recall@1"]
    for result in results
])

recall_at_3 = np.mean([
    result["recall@3"]
    for result in results
])

recall_at_5 = np.mean([
    result["recall@5"]
    for result in results
])

recall_at_10 = np.mean([
    result["recall@10"]
    for result in results
])


# -----------------------------
# Print results
# -----------------------------
print("\n" + "=" * 60)
print("RETRIEVAL EVALUATION RESULTS")
print("=" * 60)

print(f"Number of queries : {len(results)}")
print(f"Recall@1          : {recall_at_1:.3f}")
print(f"Recall@3          : {recall_at_3:.3f}")
print(f"Recall@5          : {recall_at_5:.3f}")
print(f"Recall@10         : {recall_at_10:.3f}")

print("=" * 60)


# -----------------------------
# Show individual results
# -----------------------------
print("\nINDIVIDUAL QUERY RESULTS")
print("=" * 60)

for result in results:

    print(f"\nQuery: {result['query']}")
    print(f"Expected: {result['relevant']}")

    print("Retrieved:")
    for rank, verse_id in enumerate(result["retrieved"][:5], start=1):
        print(f"  {rank}. {verse_id}")

    print(
        f"Recall@1={result['recall@1']} | "
        f"Recall@3={result['recall@3']} | "
        f"Recall@5={result['recall@5']} | "
        f"Recall@10={result['recall@10']}"
    )