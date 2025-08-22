# Football Predictor - Bundesliga Data Fetcher

## Overview

This project fetches and processes Bundesliga football data using modern Python libraries for improved performance and efficiency.

## Recent Improvements

### 1. **Modern Dependencies**
- **Polars**: Replaced pandas for faster DataFrame operations
- **httpx**: Replaced requests for async HTTP requests with better performance

### 2. **Data Storage**
- **Parquet Format**: All data is now saved as Parquet files instead of CSV for better compression and faster I/O
- **Organized Structure**: 
  - Results: `data/results/`
  - Fixtures: `data/fixtures/`

### 3. **Configuration Management**
- **JSON Config**: Current season is now stored in `config.json`
- **Centralized Settings**: Easy to update season without code changes

### 4. **Enhanced Functionality**
- **Async Support**: All download operations are now asynchronous for better performance
- **Comprehensive Coverage**: Downloads both fixtures and results for all available seasons
- **Error Handling**: Robust error handling and graceful failures
- **Type Hints**: Full type annotations for better code maintainability

## File Structure

```
football_predictor/
├── config.json                 # Configuration file with current season
├── getbundesligadata.py        # Main data fetcher class
├── example_usage.py            # Example usage script
├── pyproject.toml             # Project dependencies
├── data/
│   ├── results/               # Match results (parquet files)
│   └── fixtures/              # Fixture data (parquet files)
```

## Usage

### Basic Usage

```python
from getbundesligadata import GetBundesligaData
import asyncio

async def main():
    # Initialize with config from config.json
    downloader = GetBundesligaData()
    
    # Download all fixtures and results
    fixtures = await downloader.download_all_fixtures(years_back=5)
    results = await downloader.download_all_results(years_back=5)
    
    # Load historical data
    historical_df = downloader.get_historical_results(seasons_to_keep=3)
    fixtures_df = downloader.get_fixtures_data(seasons_to_keep=2)

# Run async code
asyncio.run(main())
```

### Configuration

Update the current season in `config.json`:

```json
{
  "current_season": "2024/2025"
}
```

### Running Examples

```bash
# Run the example script
python example_usage.py
```

## Key Methods

### Async Methods (for downloading)
- `download_all_fixtures(years_back=10)`: Download fixtures for multiple seasons
- `download_all_results(years_back=10)`: Download results for multiple seasons
- `get_fixtures_for_season(season)`: Download fixtures for a specific season
- `get_results_for_season(season)`: Download results for a specific season

### Sync Methods (for loading existing data)
- `get_historical_results(seasons_to_keep=5)`: Load historical match results
- `get_fixtures_data(seasons_to_keep=5)`: Load fixture data

## Performance Improvements

1. **Polars vs Pandas**: 2-10x faster DataFrame operations
2. **Parquet vs CSV**: 50-80% smaller file sizes, faster read/write
3. **Async Downloads**: Concurrent requests for faster data fetching
4. **Efficient Memory Usage**: Lazy evaluation and optimized data types

## Dependencies

- Python >=3.13
- polars >=1.32.3
- httpx >=0.27.0

Install with:
```bash
uv sync
```
