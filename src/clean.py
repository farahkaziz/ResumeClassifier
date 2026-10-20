import pandas as pd
import re
from html import unescape


# ============================================
# 1. Load the original dataset
# ============================================

input_path = "data/raw/Resume.csv"
output_path = "data/processed/Resume_clean.csv"

df = pd.read_csv(input_path)

print("Original dataset loaded successfully.")
print(f"Number of rows: {len(df)}")


# ============================================
# 2. Keep only the columns needed for ML
# ============================================

df = df[["ID", "Resume_str", "Category"]]


# ============================================
# 3. Function to clean resume text
# ============================================

def clean_resume(text):
    """
    Clean the resume text while preserving
    meaningful information for classification.
    """

    # Make sure the value is a string
    text = str(text)

    # Decode HTML entities such as &amp; or &#39;
    text = unescape(text)

    # Remove HTML tags if there are any
    text = re.sub(r"<[^>]+>", " ", text)

    # Replace line breaks and tabs with spaces
    text = re.sub(r"[\n\r\t]+", " ", text)

    # Remove repeated underscores
    text = re.sub(r"_+", " ", text)

    # Remove excessive spaces
    text = re.sub(r"\s+", " ", text)

    # Remove spaces at the beginning and end
    text = text.strip()

    return text


# ============================================
# 4. Apply cleaning
# ============================================

df["Resume_str"] = df["Resume_str"].apply(clean_resume)


# ============================================
# 5. Remove empty resumes
# ============================================

before = len(df)

df = df[df["Resume_str"].str.len() > 0]

removed_empty = before - len(df)

print(f"Empty resumes removed: {removed_empty}")


# ============================================
# 6. Remove duplicate resumes
# ============================================

before = len(df)

df = df.drop_duplicates(subset=["Resume_str"])

removed_duplicates = before - len(df)

print(f"Duplicate resumes removed: {removed_duplicates}")


# ============================================
# 7. Reset the index
# ============================================

df = df.reset_index(drop=True)


# ============================================
# 8. Save the cleaned dataset
# ============================================

df.to_csv(output_path, index=False)

print("\nCleaning completed successfully.")
print(f"Cleaned dataset saved to: {output_path}")
print(f"Final number of rows: {len(df)}")


# ============================================
# 9. Display a sample
# ============================================

print("\nSample of cleaned data:")
print(df.head())

print("\nCategories:")
print(df["Category"].value_counts())