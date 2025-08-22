#!/usr/bin/env python3
"""
Script to create clean parquet files from raw fixtures and results data.

This script:
1. Creates a clean season schedule from fixtures with match day, home team, away team
2. Creates a clean history from results with match day (every 9 games), home/away teams, goals
3. Applies team name mapping to normalize team names
4. Saves both as parquet files in the clean data folders
"""

import json
import os
from pathlib import Path

import pandas as pd


def load_team_mapping(mapping_file: str) -> dict:
    """Load team name mapping from JSON file."""
    with open(mapping_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("team_name_mapping", {})


def normalize_team_name(team_name: str, mapping: dict) -> str:
    """
    Normalize team name using the mapping.
    If team name is in mapping, return the mapped value.
    Otherwise, return the original name.
    """
    return mapping.get(team_name, team_name)


def process_fixtures(fixtures_file: str, team_mapping: dict, output_file: str):
    """
    Process fixtures data to create clean season schedule.

    Args:
        fixtures_file: Path to raw fixtures CSV file
        team_mapping: Dictionary for team name normalization
        output_file: Path to save clean parquet file
    """
    print(f"Processing fixtures from {fixtures_file}")

    # Read fixtures data
    df = pd.read_csv(fixtures_file)

    # Select and rename columns
    clean_df = df[["Round Number", "Date", "Home Team", "Away Team"]].copy()
    clean_df.columns = ["match_day", "date", "home_team", "away_team"]

    # Normalize team names
    clean_df["home_team"] = clean_df["home_team"].apply(
        lambda x: normalize_team_name(x, team_mapping)
    )
    clean_df["away_team"] = clean_df["away_team"].apply(
        lambda x: normalize_team_name(x, team_mapping)
    )

    # Convert date to datetime
    clean_df["date"] = pd.to_datetime(clean_df["date"], format="%d/%m/%Y %H:%M")

    # Sort by match day and date
    clean_df = clean_df.sort_values(["match_day", "date"]).reset_index(drop=True)

    # Save as parquet
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    clean_df.to_parquet(output_file, index=False)

    print(f"Saved clean fixtures to {output_file}")
    print(f"Shape: {clean_df.shape}")
    print(f"Match days: {clean_df['match_day'].min()} to {clean_df['match_day'].max()}")
    print()


def process_results(results_files: list, team_mapping: dict, output_file: str):
    """
    Process results data from multiple seasons to create clean history.

    Args:
        results_files: List of paths to raw results CSV files
        team_mapping: Dictionary for team name normalization
        output_file: Path to save clean parquet file
    """
    print(f"Processing results from {len(results_files)} files")

    all_results = []

    for results_file in results_files:
        print(f"  - Processing {os.path.basename(results_file)}")

        # Read results data
        df = pd.read_csv(results_file)

        # Select relevant columns
        season_df = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].copy()
        season_df.columns = [
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
        ]

        # Convert date to datetime
        season_df["date"] = pd.to_datetime(season_df["date"], format="%d/%m/%Y")

        # Sort by date to ensure proper chronological order
        season_df = season_df.sort_values("date").reset_index(drop=True)

        # Calculate match day (every 9 games is one match day)
        # Assuming 18 teams in Bundesliga, each match day has 9 games
        season_df["match_day"] = (season_df.index // 9) + 1

        # Extract season from filename for tracking
        filename = os.path.basename(results_file)
        season = filename.replace("_results.csv", "").replace("_", "/")
        season_df["season"] = season

        all_results.append(season_df)

    # Combine all seasons
    combined_df = pd.concat(all_results, ignore_index=True)

    # Normalize team names
    combined_df["home_team"] = combined_df["home_team"].apply(
        lambda x: normalize_team_name(x, team_mapping)
    )
    combined_df["away_team"] = combined_df["away_team"].apply(
        lambda x: normalize_team_name(x, team_mapping)
    )

    # Select final columns
    clean_df = combined_df[
        [
            "match_day",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "season",
            "date",
        ]
    ]

    # Sort by season and match day
    clean_df = clean_df.sort_values(["season", "match_day", "date"]).reset_index(
        drop=True
    )

    # Save as parquet
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    clean_df.to_parquet(output_file, index=False)

    print(f"Saved clean results to {output_file}")
    print(f"Shape: {clean_df.shape}")
    print(f"Seasons: {clean_df['season'].unique()}")
    print(f"Total match days: {clean_df['match_day'].max()}")
    print()


def main():
    """Main function to process both fixtures and results data."""

    # Define paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"

    # Team mapping file
    mapping_file = data_dir / "team_name_mapping.json"

    # Input files
    fixtures_file = data_dir / "fixtures" / "raw" / "2025_2026_fixtures.csv"
    results_dir = data_dir / "results" / "raw"

    # Output files
    clean_fixtures_file = (
        data_dir / "fixtures" / "clean" / "2025_2026_season_schedule.parquet"
    )
    clean_results_file = data_dir / "results" / "clean" / "historical_results.parquet"

    # Load team name mapping
    print("Loading team name mapping...")
    team_mapping = load_team_mapping(mapping_file)
    print(f"Loaded mapping for {len(team_mapping)} teams")
    print()

    # Process fixtures
    if fixtures_file.exists():
        process_fixtures(str(fixtures_file), team_mapping, str(clean_fixtures_file))
    else:
        print(f"Fixtures file not found: {fixtures_file}")

    # Find all results files
    results_files = list(results_dir.glob("*_results.csv"))
    results_files.sort()  # Sort to ensure consistent order

    if results_files:
        process_results(
            [str(f) for f in results_files], team_mapping, str(clean_results_file)
        )
    else:
        print(f"No results files found in: {results_dir}")

    print("Data processing complete!")


if __name__ == "__main__":
    main()
