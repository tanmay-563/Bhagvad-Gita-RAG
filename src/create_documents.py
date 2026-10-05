import pandas as pd
from pathlib import Path


# Load cleaned dataset
input_path = Path("data/bhagavad_gita_clean.csv")
output_path = Path("data/gita_documents.csv")

df = pd.read_csv(input_path)


def create_document(row):
    return (
        f"Bhagavad Gita - Chapter {row['chapter']}, "
        f"Verse {row['verse']}\n\n"
        f"{row['english']}"
    )


# Create retrieval document
df["document"] = df.apply(create_document, axis=1)


# Keep the information we'll need later
documents = df[
    [
        "verse_id",
        "chapter",
        "verse",
        "document",
        "sanskrit",
        "transliteration",
        "hindi",
        "english"
    ]
]


# Save
documents.to_csv(output_path, index=False)


print("Documents created successfully!")
print(f"Number of documents: {len(documents)}")
print(f"Saved to: {output_path}")

print("\nExample document:")
print(documents.iloc[0]["document"])