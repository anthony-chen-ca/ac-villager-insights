"""
recommender.py

Very bare-bones for now, I'll make it interactive soon.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from scripts.config import *
from scripts.file_locations import MERGED_CSV
from scripts.featurize import build_features

# PARAMETERS
selected_villagers = ["Chai", "Chelsea", "Rilla"]
top_k = 10

config = ModelConfig(
    categorical_settings=CATEGORICAL_FEATURE_LIST,
    visual_settings=[VisualFeatureConfig(name=VisualType.ICON, pca=8),
                     VisualFeatureConfig(name=VisualType.PHOTO, pca=8)],
    model_settings=ModelSettings(model=ModelType.RIDGE, alpha=1.0)
)

# LOAD DATA
df = pd.read_csv(MERGED_CSV)
X, _, feature_names = build_features(config)

# Track villager names
names = df["Name"].values
name_to_idx = {name: idx for idx, name in enumerate(names)}

# Validate selected villagers
selected_indices = [name_to_idx[name] for name in selected_villagers if name in name_to_idx]
if not selected_indices:
    raise ValueError("None of the selected villagers were found in the dataset.")

# Compute mean feature vector
query_vector = np.mean(X[selected_indices], axis=0, keepdims=True)

# Calculate cosine similarities
similarities = cosine_similarity(query_vector, X)[0]

# Create results
results = pd.DataFrame({"Name": names, "Similarity": similarities})

# Exclude selected villagers
results = results[~results["Name"].isin(selected_villagers)]

# Sort by similarity
results_sorted = results.sort_values("Similarity", ascending=False).head(top_k)

# Display recommendations
print("=== Villager Recommendations ===")
print("Based on your favorites:", ", ".join(selected_villagers))
print()
for _, row in results_sorted.iterrows():
    print(f"{row['Name']}: similarity {row['Similarity']:.4f}")
