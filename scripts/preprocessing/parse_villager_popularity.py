"""
parse_villager_popularity.py

The source for popularity data has a really weird CSV format. This file fixes said format,
merges all popularity data across dates, and stores the result in POPULARITY_CSV.
"""

import os
import pandas as pd
from scripts.file_locations import POPULARITY_ARCHIVE_DIR, POPULARITY_CSV


def extract_name_vote_pairs(filepath):
    df = pd.read_csv(filepath, header=None)
    name_vote_pairs = []

    for i in range(0, df.shape[1], 2):
        names = df.iloc[:, i]
        votes = df.iloc[:, i + 1] if i + 1 < df.shape[1] else pd.Series([None] * len(names))

        for name, vote in zip(names, votes):
            if pd.notna(name) and pd.notna(vote):
                try:
                    name = str(name).strip()
                    vote = int(vote)
                    name_vote_pairs.append((name, vote))
                except ValueError:
                    continue

    return name_vote_pairs


def aggregate_votes_from_archive():
    vote_totals = {}

    for filename in os.listdir(POPULARITY_ARCHIVE_DIR):
        if filename.endswith(".csv"):
            path = os.path.join(POPULARITY_ARCHIVE_DIR, filename)
            print(f"Processing: {filename}")
            pairs = extract_name_vote_pairs(path)

            for name, vote in pairs:
                vote_totals[name] = vote_totals.get(name, 0) + vote

    # Convert to DataFrame and save
    df_total = pd.DataFrame(sorted(vote_totals.items()), columns=["Name", "Votes"])
    df_total.to_csv(POPULARITY_CSV, index=False)
    print(f"Saved cleaned popularity data to: {POPULARITY_CSV}")


if __name__ == "__main__":
    aggregate_votes_from_archive()
