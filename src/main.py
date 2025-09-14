"""
Simple main script to run the football prediction pipeline.
"""

from advanced_prediction_model import main as predict_matches
from get_data import main as download_data
from process_match_data import main as process_data


def main():
    """Run the complete football prediction pipeline."""
    print("Football Predictor - Starting pipeline...")

    print("\n1. Downloading data...")
    download_data()

    print("\n2. Processing data...")
    process_data()

    print("\n3. Generating predictions...")
    predict_matches()

    print("\nPipeline completed! Check for prediction files in this directory.")


if __name__ == "__main__":
    main()
