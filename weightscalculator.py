import numpy as np


class WeightsCalculator:
    def __init__(self, normalized_df):
        self.normalized_df = normalized_df

    def get_weights(self, method="linear", decay_factor=-0.1):
        if method == "linear":
            seasons = self.normalized_df["season"].unique()
            seasons_normalized = [int(season.split("/")[1]) for season in seasons]

            min_season, max_season = min(seasons_normalized), max(seasons_normalized)
            linear_weights = {
                season: (season_norm - min_season) / (max_season - min_season)
                for season, season_norm in zip(seasons, seasons_normalized)
            }

            self.normalized_df["weight"] = self.normalized_df["season"].map(
                linear_weights
            )

        elif method == "season":
            seasons = self.normalized_df["season"].unique()
            seasons_normalized = [int(season.split("/")[1]) for season in seasons]

            max_season = max(seasons_normalized)
            exponential_weights = {
                season: np.exp(decay_factor * (max_season - season_norm))
                for season, season_norm in zip(seasons, seasons_normalized)
            }

            self.normalized_df["weight"] = self.normalized_df["season"].map(
                exponential_weights
            )

        elif method == "matchday":
            self.normalized_df["matchday"] = (
                self.normalized_df.groupby("season").cumcount() // 9
            ) + 1
            self.normalized_df["matchday"] = self.normalized_df["matchday"].apply(
                lambda x: f"{x:02d}"
            )
            self.normalized_df["season_matchday_id"] = (
                self.normalized_df["season"].str[-4:] + self.normalized_df["matchday"]
            ).astype(int)

            max_season_matchday_id = self.normalized_df["season_matchday_id"].max()
            self.normalized_df["weight"] = np.exp(
                decay_factor
                * (max_season_matchday_id - self.normalized_df["season_matchday_id"])
            )
            self.normalized_df["weight"] /= self.normalized_df["weight"].max()

        return self.normalized_df
