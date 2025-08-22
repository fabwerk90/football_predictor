import json
from pathlib import Path

import httpx


class BundesligaDataDownloader:
    """Simple downloader for Bundesliga results and fixtures."""

    def __init__(self):
        """Initialize with basic directory structure."""
        self.data_dir = Path("data")
        self.results_dir = self.data_dir / "results/raw"
        self.fixtures_dir = self.data_dir / "fixtures/raw"

    def download_results(self, season: str) -> bool:
        """
        Download match results for a season.

        Args:
            season: Season in format "YYYY/YYYY" (e.g., "2023/2024")

        Returns:
            bool: True if successful
        """
        start_year, end_year = season.split("/")
        short_season = f"{start_year[2:]}{end_year[2:]}"  # "2324"

        url = f"https://www.football-data.co.uk/mmz4281/{short_season}/D1.csv"
        output_path = self.results_dir / f"{season.replace('/', '_')}_results.csv"

        return self._download_file(url, output_path)

    def download_fixtures(self, season: str) -> bool:
        """
        Download fixtures for a season.

        Args:
            season: Season in format "YYYY/YYYY" (e.g., "2023/2024")

        Returns:
            bool: True if successful
        """
        start_year = season.split("/")[0]
        url = f"https://fixturedownload.com/download/bundesliga-{start_year}-WEuropeStandardTime.csv"
        output_path = self.fixtures_dir / f"{season.replace('/', '_')}_fixtures.csv"

        return self._download_file(url, output_path)

    def _download_file(self, url: str, output_path: Path) -> bool:
        """Download a file from URL and save it."""
        try:
            response = httpx.get(url)
            response.raise_for_status()

            output_path.write_bytes(response.content)
            print(f"Downloaded: {output_path}")
            return True

        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return False


def main():
    """Download data for current and recent seasons."""
    downloader = BundesligaDataDownloader()

    # Download last few seasons
    # Load seasons from config.json
    config_path = Path("config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    current_season = config.get("current_season", "")
    historical_results = config.get("history_list", [])

    downloader.download_fixtures(current_season)

    for season in historical_results:
        print(f"Downloading data for {season}...")
        downloader.download_results(season)


if __name__ == "__main__":
    main()
