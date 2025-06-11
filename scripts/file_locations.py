"""
file_locations.py

Locations for various directories and files.
"""

import os

# Root directory (relative to this file)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data folder
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Images folder
IMAGE_DIR = os.path.join(DATA_DIR, "images")
ICON_DIR = os.path.join(IMAGE_DIR, "icons")
PHOTO_DIR = os.path.join(IMAGE_DIR, "photos")

# Raw folder
RAW_DIR = os.path.join(DATA_DIR, "raw")
POPULARITY_ARCHIVE_DIR = os.path.join(RAW_DIR, "popularity_archive")

# Interim folder
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
VILLAGER_CSV = os.path.join(INTERIM_DIR, "villagers.csv")
POPULARITY_CSV = os.path.join(INTERIM_DIR, "villager_popularity.csv")
CLIP_EMBEDDINGS_CSV = os.path.join(INTERIM_DIR, "villager_clip_embeddings.csv")
POCKET_CAMP_THEME_CSV = os.path.join(INTERIM_DIR, "pocket_camp_themes.csv")
CUSTOM_TAG_CSV = os.path.join(INTERIM_DIR, "villager_custom_tags.csv")

# Processed folder
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MERGED_CSV = os.path.join(PROCESSED_DIR, "villagers_merged.csv")

# Outputs folder
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
HYPERPARAMETER_RESULTS_CSV = os.path.join(OUTPUTS_DIR, "hyperparameter_sweep_results.csv")

# Scripts folder
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
MODELING_DIR = os.path.join(SCRIPTS_DIR, "modeling")
PREPROCESSING_DIR = os.path.join(SCRIPTS_DIR, "preprocessing")
