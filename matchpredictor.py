import numpy as np
import pandas as pd
from config import CURRENT_SEASON
from scipy.stats import poisson


class MatchPredictor:
    def __init__(self, performance_df, current_season_avg_goals, max_goals):
        self.performance_df = performance_df
        self.current_season_avg_goals = current_season_avg_goals
        self.max_goals = max_goals

    def predict_goals(self, home_team, away_team):
        home_perf = self.performance_df[self.performance_df["team"] == home_team].iloc[
            0
        ]
        away_perf = self.performance_df[self.performance_df["team"] == away_team].iloc[
            0
        ]

        home_expected_goals = (
            home_perf["home_attack"]
            * away_perf["away_defense"]
            * self.current_season_avg_goals[CURRENT_SEASON]["home"]
        )
        away_expected_goals = (
            away_perf["away_attack"]
            * home_perf["home_defense"]
            * self.current_season_avg_goals[CURRENT_SEASON]["away"]
        )

        goal_range = np.arange(0, self.max_goals)
        home_goals = np.argmax(poisson.pmf(k=goal_range, mu=home_expected_goals))
        away_goals = np.argmax(poisson.pmf(k=goal_range, mu=away_expected_goals))

        return home_goals, away_goals

    def predict_matches(self, season_schedule, matchday):
        matchday_df = season_schedule[
            season_schedule["matchday"] == matchday
        ].reset_index(drop=True)
        predictions = matchday_df.apply(
            lambda row: self.predict_goals(row["home_team"], row["away_team"]), axis=1
        )
        matchday_df[["pred_home_goals", "pred_away_goals"]] = pd.DataFrame(
            predictions.tolist(), index=matchday_df.index
        )
        return matchday_df
