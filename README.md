# football_predictor

## Overview

The football_predictor project is designed to predict the outcomes of Bundesliga matches using historical data and statistical models. The project fetches historical match results, normalizes team names, calculates team performance metrics, and predicts match outcomes based on these metrics.

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/fabwerk90/football_predictor.git
   cd football_predictor
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To use the project, follow these steps:

1. Update the `config.py` file with the current season and the number of seasons to keep:
   ```python
   CURRENT_SEASON = "2024/2025"
   SEASONS_TO_KEEP = 2
   ```

2. Run the main script to fetch data, calculate performance metrics, and predict match outcomes:
   ```bash
   python main.py
   ```

3. The predicted match outcomes for the specified matchday will be printed to the console.

## Contribution Guidelines

We welcome contributions to the football_predictor project. To contribute, follow these steps:

1. Fork the repository on GitHub.
2. Create a new branch with a descriptive name:
   ```bash
   git checkout -b my-new-feature
   ```
3. Make your changes and commit them with clear and concise commit messages.
4. Push your changes to your forked repository:
   ```bash
   git push origin my-new-feature
   ```
5. Create a pull request on the main repository, describing the changes you have made and the reasons for them.
