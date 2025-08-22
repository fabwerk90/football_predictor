"""
Advanced football prediction model with optional extensions.
Implements Dixon-Coles correction and time-weighted analysis.
"""

from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import poisson


class AdvancedFootballPredictor:
    """
    Advanced prediction model with Dixon-Coles correction and time weighting.
    """

    def __init__(self, xi: float = 0.0054):
        """
        Initialize advanced predictor.

        Args:
            xi: Time decay parameter for weighting older matches (default from Dixon-Coles)
        """
        self.xi = xi  # Time decay parameter
        self.team_stats = {}
        self.league_avg_goals = 0.0
        self.rho = 0.0  # Dixon-Coles parameter

    def load_data(self, results_path: str, fixtures_path: str):
        """Load historical results and upcoming fixtures."""
        self.results_df = pd.read_parquet(results_path)
        self.fixtures_df = pd.read_parquet(fixtures_path)

        # Convert date columns to datetime
        self.results_df["date"] = pd.to_datetime(self.results_df["date"])

        return self.results_df, self.fixtures_df

    def calculate_time_weights(self, reference_date: datetime = None):
        """Calculate time-based weights for historical matches."""
        if reference_date is None:
            reference_date = datetime.now()

        # Calculate days since each match
        days_since = (reference_date - self.results_df["date"]).dt.days

        # Calculate exponential weights: w = exp(-xi * delta_t)
        weights = np.exp(-self.xi * days_since)

        return weights

    def calculate_team_stats_weighted(self):
        """Calculate team statistics with time weighting."""
        print("Calculating weighted team statistics...")

        # Get time weights
        weights = self.calculate_time_weights()

        # Add weights to dataframe for calculation
        df = self.results_df.copy()
        df["weight"] = weights

        # Calculate weighted league averages
        total_weighted_goals = (df["home_goals"] * df["weight"]).sum() + (
            df["away_goals"] * df["weight"]
        ).sum()
        total_weight = df["weight"].sum() * 2  # Each match has 2 teams
        self.league_avg_goals = total_weighted_goals / total_weight

        # Get all unique teams
        teams = set(df["home_team"].unique()).union(set(df["away_team"].unique()))

        # Calculate weighted stats for each team
        for team in teams:
            # Home matches
            home_matches = df[df["home_team"] == team]
            home_goals_scored = (
                home_matches["home_goals"] * home_matches["weight"]
            ).sum()
            home_goals_conceded = (
                home_matches["away_goals"] * home_matches["weight"]
            ).sum()
            home_weight_sum = home_matches["weight"].sum()

            # Away matches
            away_matches = df[df["away_team"] == team]
            away_goals_scored = (
                away_matches["away_goals"] * away_matches["weight"]
            ).sum()
            away_goals_conceded = (
                away_matches["home_goals"] * away_matches["weight"]
            ).sum()
            away_weight_sum = away_matches["weight"].sum()

            # Calculate weighted averages
            avg_home_goals_scored = (
                home_goals_scored / home_weight_sum if home_weight_sum > 0 else 0
            )
            avg_home_goals_conceded = (
                home_goals_conceded / home_weight_sum if home_weight_sum > 0 else 0
            )
            avg_away_goals_scored = (
                away_goals_scored / away_weight_sum if away_weight_sum > 0 else 0
            )
            avg_away_goals_conceded = (
                away_goals_conceded / away_weight_sum if away_weight_sum > 0 else 0
            )

            self.team_stats[team] = {
                "home_goals_scored_avg": avg_home_goals_scored,
                "home_goals_conceded_avg": avg_home_goals_conceded,
                "away_goals_scored_avg": avg_away_goals_scored,
                "away_goals_conceded_avg": avg_away_goals_conceded,
                "total_home_weight": home_weight_sum,
                "total_away_weight": away_weight_sum,
                "total_games": len(home_matches)
                + len(away_matches),  # Track total games
            }

        print(f"Weighted statistics calculated for {len(teams)} teams")
        print(
            f"Weighted league average goals per team per match: {self.league_avg_goals:.2f}"
        )

    def estimate_dixon_coles_rho(self):
        """
        Estimate Dixon-Coles rho parameter from historical data.
        This is a simplified estimation based on low-scoring game frequencies.
        """
        df = self.results_df

        # Count frequencies of low-scoring results
        total_matches = len(df)
        result_0_0 = len(df[(df["home_goals"] == 0) & (df["away_goals"] == 0)])
        result_1_1 = len(df[(df["home_goals"] == 1) & (df["away_goals"] == 1)])

        # Calculate observed frequencies
        obs_0_0 = result_0_0 / total_matches
        obs_1_1 = result_1_1 / total_matches

        # Simple estimation: rho affects the ratio of actual vs expected low scores
        # This is a heuristic approximation
        if obs_0_0 > 0.05:  # If 0-0 draws are common
            self.rho = -0.1
        elif obs_1_1 > 0.12:  # If 1-1 draws are common
            self.rho = 0.1
        else:
            self.rho = 0.0

        print(f"Estimated Dixon-Coles rho parameter: {self.rho:.3f}")

    def get_team_stats_with_fallback(self, team: str) -> Dict:
        """
        Get team statistics with fallback to reduced stats for teams with insufficient data.

        Args:
            team: Team name

        Returns:
            Dictionary with team statistics
        """
        if team in self.team_stats:
            team_data = self.team_stats[team]
            # If team has more than 5 games, use calculated stats
            if team_data["total_games"] > 5:
                return team_data
            else:
                # If team has 5 or fewer games, use fallback scenario
                print(
                    f"Warning: {team} has only {team_data['total_games']} games, using reduced stats for insufficient data"
                )
                # Use 80% of league average for goals scored, 120% for goals conceded (weaker team)
                return {
                    "home_goals_scored_avg": self.league_avg_goals * 0.8,
                    "home_goals_conceded_avg": self.league_avg_goals * 1.2,
                    "away_goals_scored_avg": self.league_avg_goals * 0.8,
                    "away_goals_conceded_avg": self.league_avg_goals * 1.2,
                    "total_home_weight": 1.0,  # Minimal weight
                    "total_away_weight": 1.0,
                    "total_games": team_data["total_games"],
                }

        # Team not found in historical data at all
        print(
            f"Warning: No historical data for {team}, using reduced stats for new team"
        )
        # Use 80% of league average for goals scored, 120% for goals conceded (weaker team)
        return {
            "home_goals_scored_avg": self.league_avg_goals * 0.8,
            "home_goals_conceded_avg": self.league_avg_goals * 1.2,
            "away_goals_scored_avg": self.league_avg_goals * 0.8,
            "away_goals_conceded_avg": self.league_avg_goals * 1.2,
            "total_home_weight": 1.0,  # Minimal weight
            "total_away_weight": 1.0,
            "total_games": 0,
        }

    def dixon_coles_adjustment(
        self, home_goals: int, away_goals: int, lambda_home: float, lambda_away: float
    ) -> float:
        """
        Apply Dixon-Coles adjustment for low-scoring games.

        Args:
            home_goals: Number of home goals
            away_goals: Number of away goals
            lambda_home: Expected home goals
            lambda_away: Expected away goals

        Returns:
            Adjusted probability
        """
        base_prob = poisson.pmf(home_goals, lambda_home) * poisson.pmf(
            away_goals, lambda_away
        )

        # Apply adjustment only for low-scoring combinations
        if (home_goals, away_goals) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            if home_goals == 0 and away_goals == 0:
                tau = 1 - lambda_home * lambda_away * self.rho
            elif home_goals == 1 and away_goals == 0:
                tau = 1 + lambda_away * self.rho
            elif home_goals == 0 and away_goals == 1:
                tau = 1 + lambda_home * self.rho
            elif home_goals == 1 and away_goals == 1:
                tau = 1 - self.rho
            else:
                tau = 1.0

            return base_prob * tau

        return base_prob

    def predict_match_advanced(self, home_team: str, away_team: str) -> Dict:
        """
        Advanced match prediction with Dixon-Coles correction.
        """
        home_stats = self.get_team_stats_with_fallback(home_team)
        away_stats = self.get_team_stats_with_fallback(away_team)

        # Calculate expected goals using weighted averages
        lambda_home = (
            home_stats["home_goals_scored_avg"] * away_stats["away_goals_conceded_avg"]
        ) / self.league_avg_goals

        lambda_away = (
            away_stats["away_goals_scored_avg"] * home_stats["home_goals_conceded_avg"]
        ) / self.league_avg_goals

        # Calculate outcome probabilities with Dixon-Coles adjustment
        max_goals = 8  # Consider up to 8 goals per team for efficiency

        home_win_prob = 0.0
        draw_prob = 0.0
        away_win_prob = 0.0

        most_likely_prob = 0.0
        most_likely_result = "0-0"

        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                # Get adjusted probability
                prob = self.dixon_coles_adjustment(
                    home_goals, away_goals, lambda_home, lambda_away
                )

                # Track most likely result
                if prob > most_likely_prob:
                    most_likely_prob = prob
                    most_likely_result = f"{home_goals}-{away_goals}"

                # Add to outcome probabilities
                if home_goals > away_goals:
                    home_win_prob += prob
                elif home_goals == away_goals:
                    draw_prob += prob
                else:
                    away_win_prob += prob

        return {
            "home_team": home_team,
            "away_team": away_team,
            "expected_home_goals": lambda_home,
            "expected_away_goals": lambda_away,
            "home_win_probability": home_win_prob,
            "draw_probability": draw_prob,
            "away_win_probability": away_win_prob,
            "most_likely_result": most_likely_result,
            "most_likely_probability": most_likely_prob,
            "dixon_coles_rho": self.rho,
            "time_weighted": True,
        }

    def predict_matchday(self, matchday: int) -> List[Dict]:
        """Predict all matches for a specific matchday."""
        matchday_fixtures = self.fixtures_df[self.fixtures_df["match_day"] == matchday]
        predictions = []

        print(f"\nGenerating advanced predictions for Matchday {matchday}...")
        print(f"Found {len(matchday_fixtures)} matches")

        for _, fixture in matchday_fixtures.iterrows():
            prediction = self.predict_match_advanced(
                fixture["home_team"], fixture["away_team"]
            )
            predictions.append(prediction)

        return predictions

    def print_predictions(self, predictions: List[Dict]):
        """Print predictions in a readable format."""
        print(f"\n{'=' * 70}")
        print("FOOTBALL MATCH PREDICTIONS")
        print(f"{'=' * 70}")

        for pred in predictions:
            print(f"{pred['home_team']} vs {pred['away_team']}")
            print(
                f"Expected Goals: {pred['expected_home_goals']:.2f} - {pred['expected_away_goals']:.2f}"
            )
            print("-" * 50)

    def save_predictions(self, predictions: List[Dict], filename: str):
        """Save predictions to CSV file with only team names and expected goals."""
        if predictions:
            # Create simplified dataframe with only required columns
            simplified_predictions = []
            for pred in predictions:
                simplified_predictions.append(
                    {
                        "home_team": pred["home_team"],
                        "away_team": pred["away_team"],
                        "expected_home_goals": pred["expected_home_goals"],
                        "expected_away_goals": pred["expected_away_goals"],
                    }
                )

            predictions_df = pd.DataFrame(simplified_predictions)
            predictions_df.to_csv(filename, index=False)
            print(f"\nSimplified predictions saved to: {filename}")


def main():
    """Main function to run advanced predictions."""
    print("Advanced Football Predictor - Starting...")
    print("Features: Dixon-Coles correction + Time weighting")

    # Initialize predictor
    predictor = AdvancedFootballPredictor(xi=0.0054)  # Standard time decay

    # Load data
    results_path = "data/results/clean/historical_results.parquet"
    fixtures_path = "data/fixtures/clean/2025_2026_season_schedule.parquet"

    print("\nLoading data...")
    results_df, fixtures_df = predictor.load_data(results_path, fixtures_path)
    print(f"Loaded {len(results_df)} historical matches")
    print(f"Loaded {len(fixtures_df)} upcoming fixtures")

    # Calculate weighted team statistics
    predictor.calculate_team_stats_weighted()

    # Estimate Dixon-Coles parameter
    predictor.estimate_dixon_coles_rho()

    # Predict matchday 1
    predictions = predictor.predict_matchday(matchday=1)

    # Display and save predictions
    predictor.print_predictions(predictions)
    predictor.save_predictions(predictions, "matchday_1_predictions_advanced.csv")

    print("\nAdvanced prediction model completed successfully!")


if __name__ == "__main__":
    main()
