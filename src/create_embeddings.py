import pandas as pd
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


# ==========================================
# 1. Load documents
# ==========================================

df = pd.read_csv("data/gita_documents.csv")

documents = df["document"].tolist()

print(f"Loaded {len(documents)} documents")


# ==========================================
# 2. Load embedding model
# ==========================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded successfully")


# ==========================================
# 3. Generate embeddings
# ==========================================

print("Generating embeddings...")

embeddings = model.encode(
    documents,
    show_progress_bar=True
)

embeddings = np.array(embeddings).astype("float32")

print("Embeddings generated!")
print("Embedding shape:", embeddings.shape)


# ==========================================
# 4. Normalize embeddings
# ==========================================

faiss.normalize_L2(embeddings)


# ==========================================
# 5. Create FAISS index
# ==========================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)


# ==========================================
# 6. Save FAISS index
# ==========================================

faiss.write_index(
    index,
    "data/gita.index"
)


# ==========================================
# 7. Save metadata
# ==========================================

df.to_pickle("data/gita_metadata.pkl")


print("\n================================")
print("Embedding pipeline completed!")
print("================================")

print("Number of vectors:", index.ntotal)
print("Vector dimensions:", dimension)
print("FAISS index saved to: data/gita.index")
print("Metadata saved to: data/gita_metadata.pkl")