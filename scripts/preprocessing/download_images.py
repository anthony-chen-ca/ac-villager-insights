"""
download_images.py

Downloads villager icon and photo images from VILLAGER_CSV. Stores them in ICON_DIR and PHOTO_DIR.
"""

import os
import pandas as pd
import requests
from tqdm import tqdm
from scripts.config import ICON_DIR, PHOTO_DIR, VILLAGER_CSV

# Ensure output folders exist
os.makedirs(ICON_DIR, exist_ok=True)
os.makedirs(PHOTO_DIR, exist_ok=True)

# Load the CSV
df = pd.read_csv(VILLAGER_CSV)


def clean_filename(filename):
    return filename.strip().replace(" ", "_").replace("/", "-")


def download_image(url, filepath):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(response.content)
    except Exception as e:
        print(f"Failed to download {url} -> {filepath}: {e}")


# Loop over villagers
for _, row in tqdm(df.iterrows(), total=len(df)):
    name = clean_filename(row["Name"])
    icon_url = row.get("Icon Image URL")
    photo_url = row.get("Photo Image URL")

    if pd.notna(icon_url):
        icon_path = os.path.join(ICON_DIR, f"{name}_Icon.png")
        if not os.path.exists(icon_path):
            download_image(icon_url, icon_path)

    if pd.notna(photo_url):
        photo_path = os.path.join(PHOTO_DIR, f"{name}_Photo.png")
        if not os.path.exists(photo_path):
            download_image(photo_url, photo_path)
