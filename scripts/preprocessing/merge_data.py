"""
merge_data.py

Merges VILLAGER_CSV, POPULARITY_CSV, CLIP_EMBEDDINGS_CSV into MERGED_CSV. The final merged
dataset will be used for downstream tasks.
"""

import pandas as pd
from scripts.config import VILLAGER_CSV, POPULARITY_CSV, CLIP_EMBEDDINGS_CSV, MERGED_CSV

# Load all data
df_villagers = pd.read_csv(VILLAGER_CSV)
df_votes = pd.read_csv(POPULARITY_CSV)
df_clip = pd.read_csv(CLIP_EMBEDDINGS_CSV)

# Merge them step-by-step on 'Name'
df_merged = df_villagers.merge(df_clip, on="Name", how="inner")
df_merged = df_merged.merge(df_votes, on="Name", how="inner")

# Sanity check
print(f"Merged dataset has {len(df_merged)} rows and {df_merged.shape[1]} columns.")

# Save the final merged dataset
df_merged.to_csv(MERGED_CSV, index=False)
print(f"Saved merged data to: {MERGED_CSV}")
