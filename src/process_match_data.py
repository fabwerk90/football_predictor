"""
Script to process and merge match data from JSON files into clean parquet format.
Separates historical results from upcoming fixtures.
"""

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_json_data(file_path: str) -> List[Dict]:
    """Load match data from JSON file."""
    print(f"Loading data from: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} matches")
    return data


def extract_match_info(match: Dict) -> Dict:
    """Extract relevant information from a match record."""
    # Extract team names
    home_team = match["team1"]["teamName"]
    away_team = match["team2"]["teamName"]

    # Extract date and matchday info
    match_date = match["matchDateTimeUTC"]
    match_day = match["group"]["groupOrderID"]
    season = match["leagueSeason"]

    # Initialize goals
    home_goals = None
    away_goals = None

    # Extract final result if match is finished
    if match["matchIsFinished"] and match["matchResults"]:
        # Find the final result (Endergebnis)
        for result in match["matchResults"]:
            if result["resultName"] == "Endergebnis":
                home_goals = result["pointsTeam1"]
                away_goals = result["pointsTeam2"]
                break

    return {
        "match_id": match["matchID"],
        "date": match_date,
        "season": season,
        "match_day": match_day,
        "home_team": home_team,
        "away_team": away_team,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "is_finished": match["matchIsFinished"],
    }


def process_all_match_files(data_dir: str) -> pd.DataFrame:
    """Process all JSON files and combine into one DataFrame."""
    all_matches = []

    # Get all JSON files in the raw data directory
    json_files = list(Path(data_dir).glob("*.json"))
    print(f"Found {len(json_files)} JSON files to process")

    for json_file in sorted(json_files):
        matches_data = load_json_data(str(json_file))

        for match in matches_data:
            match_info = extract_match_info(match)
            all_matches.append(match_info)

    # Create DataFrame
    df = pd.DataFrame(all_matches)
    print(f"Total matches processed: {len(df)}")

    return df


def clean_and_split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean data and split into historical results and upcoming fixtures."""

    # Convert date column to timezone-aware datetime (UTC)
    # The source timestamps are expected to be in UTC (matchDateTimeUTC).
    df["date"] = pd.to_datetime(df["date"], utc=True)

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # Split into finished and unfinished matches
    finished_matches = df[df["is_finished"]].copy()
    upcoming_fixtures = df[~df["is_finished"]].copy()

    # For finished matches, ensure we have goal data
    finished_matches = finished_matches.dropna(subset=["home_goals", "away_goals"])

    # Convert goal columns to int for finished matches
    finished_matches["home_goals"] = finished_matches["home_goals"].astype(int)
    finished_matches["away_goals"] = finished_matches["away_goals"].astype(int)

    print(f"Historical results: {len(finished_matches)} matches")
    print(f"Upcoming fixtures: {len(upcoming_fixtures)} matches")

    return finished_matches, upcoming_fixtures


def save_processed_data(results_df: pd.DataFrame, fixtures_df: pd.DataFrame):
    """Save processed data to parquet files."""

    # Create output directories
    results_dir = Path("../data/results")
    fixtures_dir = Path("../data/fixtures")

    results_dir.mkdir(parents=True, exist_ok=True)
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    # Save historical results
    results_path = results_dir / "historical_results.parquet"
    results_df.to_parquet(results_path, index=False)
    print(f"Historical results saved to: {results_path}")

    # Save upcoming fixtures (2025_2026 season)
    fixtures_path = fixtures_dir / "2025_2026_season_schedule.parquet"
    fixtures_df.to_parquet(fixtures_path, index=False)
    print(f"Upcoming fixtures saved to: {fixtures_path}")

    return str(results_path), str(fixtures_path)


def print_data_summary(results_df: pd.DataFrame, fixtures_df: pd.DataFrame):
    """Print summary of the processed data."""
    print("\n" + "=" * 60)
    print("DATA PROCESSING SUMMARY")
    print("=" * 60)

    print("\nHistorical Results:")
    print(f"  Total matches: {len(results_df)}")
    print(f"  Date range: {results_df['date'].min()} to {results_df['date'].max()}")
    print(f"  Seasons: {sorted(results_df['season'].unique())}")
    print(
        f"  Teams: {len(set(results_df['home_team'].unique()) | set(results_df['away_team'].unique()))}"
    )

    if len(fixtures_df) > 0:
        print("\nUpcoming Fixtures:")
        print(f"  Total fixtures: {len(fixtures_df)}")
        print(
            f"  Season: {fixtures_df['season'].iloc[0] if len(fixtures_df) > 0 else 'N/A'}"
        )
        print(f"  Next matchdays: {sorted(fixtures_df['match_day'].unique())[:5]}")
        print(
            f"  Teams: {len(set(fixtures_df['home_team'].unique()) | set(fixtures_df['away_team'].unique()))}"
        )

        # Show next few fixtures
        print("\nNext upcoming matches:")
        next_fixtures = fixtures_df.head(3)
        for _, fixture in next_fixtures.iterrows():
            print(
                f"  Matchday {fixture['match_day']}: {fixture['home_team']} vs {fixture['away_team']}"
            )


def main():
    """Main function to process match data."""
    print("Starting match data processing...")

    # Define paths
    raw_data_dir = "../data/raw"

    # Process all match files
    all_matches_df = process_all_match_files(raw_data_dir)

    # Clean and split data
    results_df, fixtures_df = clean_and_split_data(all_matches_df)

    # Save processed data
    results_path, fixtures_path = save_processed_data(results_df, fixtures_df)

    # Print summary
    print_data_summary(results_df, fixtures_df)

    print("\nData processing completed successfully!")
    print(f"Results file: {results_path}")
    print(f"Fixtures file: {fixtures_path}")


if __name__ == "__main__":
    main()
