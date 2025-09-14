import json
from pathlib import Path

import httpx


class OpenLigaDBDownloader:
    """Downloader for Bundesliga data from OpenLigaDB API."""

    def __init__(self):
        """Initialize with basic directory structure."""
        project_root = Path(__file__).resolve().parent.parent
        self.data_dir = project_root / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.base_url = "https://api.openligadb.de"

    def download_season_data(self, season: str) -> bool:
        """
        Download all match data for a season (both fixtures and results).

        Args:
            season: Season in format "YYYY/YYYY" (e.g., "2023/2024")

        Returns:
            bool: True if successful
        """
        # Extract the start year from the season string
        start_year = season.split("/")[0]

        # OpenLigaDB uses the start year to identify the season
        url = f"{self.base_url}/getmatchdata/bl1/{start_year}"
        output_path = self.raw_data_dir / f"{season.replace('/', '_')}_matches.json"

        return self._download_json(url, output_path)

    def _download_json(self, url: str, output_path: Path) -> bool:
        """Download JSON data from URL and save it."""
        try:
            # Use a reasonable timeout so the call fails fast on network issues
            response = httpx.get(url, timeout=15.0)
            response.raise_for_status()

            # Parse JSON to validate it
            data = response.json()

            # Ensure the target directory exists before writing
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save as formatted JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"Downloaded {len(data)} matches for season: {output_path}")
            return True

        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return False


def main():
    """Download data for current and recent seasons."""
    downloader = OpenLigaDBDownloader()

    # Load seasons from config.json
    config_path = Path("config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    current_season = config.get("current_season", "")
    historical_seasons = config.get("history_list", [])

    # Download data for all seasons (both historical and current)
    all_seasons = set(historical_seasons)
    if current_season:
        all_seasons.add(current_season)

    for season in sorted(all_seasons):
        print(f"Downloading data for {season}...")
        success = downloader.download_season_data(season)
        if not success:
            print(f"Failed to download data for {season}")


if __name__ == "__main__":
    main()
