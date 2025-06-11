"""
merge_data.py

Merges the following:
- VILLAGER_CSV
- POPULARITY_CSV
- CLIP_EMBEDDINGS_CSV
- POCKET_CAMP_THEME_CSV
- CUSTOM_TAG_CSV
into MERGED_CSV. The final merged dataset will be used for downstream tasks.
"""

import pandas as pd
from scripts.file_locations import (VILLAGER_CSV,
                                    POPULARITY_CSV,
                                    CLIP_EMBEDDINGS_CSV,
                                    POCKET_CAMP_THEME_CSV,
                                    CUSTOM_TAG_CSV,
                                    MERGED_CSV)

# Load all data
df_villagers = pd.read_csv(VILLAGER_CSV)
df_votes = pd.read_csv(POPULARITY_CSV)
df_clip = pd.read_csv(CLIP_EMBEDDINGS_CSV)
df_theme = pd.read_csv(POCKET_CAMP_THEME_CSV)
df_custom = pd.read_csv(CUSTOM_TAG_CSV)

# Merge them step-by-step on 'Name'
df_merged = df_villagers.merge(df_theme, on="Name", how="inner")
df_merged = df_merged.merge(df_custom, on="Name", how="inner")
df_merged = df_merged.merge(df_clip, on="Name", how="inner")
df_merged = df_merged.merge(df_votes, on="Name", how="inner")

# Sanity check
print(f"Merged dataset has {len(df_merged)} rows and {df_merged.shape[1]} columns.")

# Save the final merged dataset
df_merged.to_csv(MERGED_CSV, index=False)
print(f"Saved merged data to: {MERGED_CSV}")
