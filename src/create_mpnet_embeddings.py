import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer


# -----------------------------
# File paths
# -----------------------------
DOCUMENT_FILE = "data/gita_documents.csv"
INDEX_FILE = "data/gita_mpnet.index"
METADATA_FILE = "data/gita_mpnet_metadata.pkl"


# -----------------------------
# Load documents
# -----------------------------
df = pd.read_csv(DOCUMENT_FILE)

print(f"Loaded {len(df)} documents.")


# -----------------------------
# Load embedding model
# -----------------------------
model_name = "all-mpnet-base-v2"

print(f"Loading model: {model_name}")

model = SentenceTransformer(model_name)


# -----------------------------
# Generate embeddings
# -----------------------------
documents = df["document"].tolist()

embeddings = model.encode(
    documents,
    show_progress_bar=True,
    convert_to_numpy=True
)


print(f"Embedding shape: {embeddings.shape}")


# -----------------------------
# Normalize embeddings
# -----------------------------
embeddings = embeddings.astype("float32")

faiss.normalize_L2(embeddings)


# -----------------------------
# Create FAISS index
# -----------------------------
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)


# -----------------------------
# Save index
# -----------------------------
faiss.write_index(index, INDEX_FILE)


# -----------------------------
# Save metadata
# -----------------------------
with open(METADATA_FILE, "wb") as f:
    pickle.dump(df, f)


# -----------------------------
# Final information
# -----------------------------
print("\nMPNet embeddings generated successfully!")

print(f"Embedding dimensions: {dimension}")
print(f"Number of vectors: {index.ntotal}")

print(f"FAISS index saved to: {INDEX_FILE}")
print(f"Metadata saved to: {METADATA_FILE}")