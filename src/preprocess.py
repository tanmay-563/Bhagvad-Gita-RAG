import pandas as pd
from pathlib import Path


# =========================
# 1. Load raw dataset
# =========================

input_path = Path("data/bhagavad_gita.csv")
output_path = Path("data/bhagavad_gita_clean.csv")

df = pd.read_csv(input_path)


# =========================
# 2. Clean text columns
# =========================

text_columns = [
    "sanskrit",
    "hindi",
    "english",
    "transliteration"
]

for column in text_columns:
    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


# =========================
# 3. Create unique verse ID
# =========================

df["verse_id"] = (
    "BG_"
    + df["chapter"].astype(str)
    + "_"
    + df["verse"].astype(str)
)


# =========================
# 4. Reorder columns
# =========================

df = df[
    [
        "verse_id",
        "chapter",
        "verse",
        "sanskrit",
        "transliteration",
        "hindi",
        "english"
    ]
]


# =========================
# 5. Remove exact duplicates
# =========================

df = df.drop_duplicates(subset=["verse_id"])


# =========================
# 6. Sort by chapter and verse
# =========================

df = df.sort_values(
    by=["chapter", "verse"]
).reset_index(drop=True)


# =========================
# 7. Save cleaned dataset
# =========================

df.to_csv(output_path, index=False)


# =========================
# 8. Validation
# =========================

print("Cleaned dataset created successfully!")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Saved to: {output_path}")

print("\nFirst 5 records:")
print(df.head())

print("\nLast 5 records:")
print(df.tail())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate verse IDs:")
print(df["verse_id"].duplicated().sum())