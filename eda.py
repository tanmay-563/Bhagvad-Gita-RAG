import pandas as pd

# Load dataset
df = pd.read_csv("data/bhagavad_gita.csv")

# Basic information
print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMNS ==========")
print(df.columns.tolist())

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

print("\n========== UNIQUE CHAPTERS ==========")
print(df["chapter"].unique())

print("\n========== VERSES PER CHAPTER ==========")
print(df.groupby("chapter")["verse"].count())

print("\n========== BASIC STATISTICS ==========")
print(df.describe(include="all"))