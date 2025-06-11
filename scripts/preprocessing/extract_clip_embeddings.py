"""
extract_clip_embeddings.py

Extracts CLIP embeddings from villager images using OpenAI's CLIP model.
Embeddings are saved as CSV files for use in downstream tasks.
"""

import os
import pandas as pd
import torch
import clip
from PIL import Image
from tqdm import tqdm
from scripts.file_locations import ICON_DIR, PHOTO_DIR, VILLAGER_CSV, CLIP_EMBEDDINGS_CSV
from scripts.config import VisualType

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load villager names
df = pd.read_csv(VILLAGER_CSV)
names = df["Name"].dropna().unique()

# Embedding storage
rows = []

for name in tqdm(names):
    name_clean = name.strip().replace(" ", "_")

    # File paths
    icon_path = os.path.join(ICON_DIR, f"{name_clean}_Icon.png")
    photo_path = os.path.join(PHOTO_DIR, f"{name_clean}_Photo.png")

    row = {"Name": name}

    # Process Icon
    if os.path.exists(icon_path):
        try:
            image = preprocess(Image.open(icon_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                icon_embed = model.encode_image(image).cpu().squeeze().numpy()
            for i, val in enumerate(icon_embed):
                row[f"{VisualType.ICON.value} CLIP {i}"] = val
        except Exception as e:
            print(f"Failed to process icon for {name}: {e}")

    # Process Photo
    if os.path.exists(photo_path):
        try:
            image = preprocess(Image.open(photo_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                photo_embed = model.encode_image(image).cpu().squeeze().numpy()
            for i, val in enumerate(photo_embed):
                row[f"{VisualType.PHOTO.value} CLIP {i}"] = val
        except Exception as e:
            print(f"Failed to process photo for {name}: {e}")

    rows.append(row)

# Create DataFrame
df_embed = pd.DataFrame(rows)
df_embed.to_csv(CLIP_EMBEDDINGS_CSV, index=False)
print(f"Saved CLIP embeddings to {CLIP_EMBEDDINGS_CSV}")
