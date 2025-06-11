"""
recommender.py

Very bare-bones for now, I'll make it interactive soon.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scripts.file_locations import MERGED_CSV
from scripts.featurize import build_features

# PARAMETERS
icon_weight = 0.5
photo_weight = 0.5
use_categorical = True
n_components_icon = 8
n_components_photo = 8
random_state = 1

selected_villagers = ["Lobo", "Kyle", "Tiansheng"]
top_k = 10

# LOAD DATA
df = pd.read_csv(MERGED_CSV)
X, _, feature_names = build_features(
    df,
    icon_weight,
    photo_weight,
    use_categorical,
    n_components_icon,
    n_components_photo,
    random_state,
)

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
