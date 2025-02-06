import numpy as np


class WeightsCalculator:
    """
    A class to calculate weights for historical match data based on different methods.

    Attributes:
        normalized_df (pd.DataFrame): DataFrame containing normalized historical match data.
    """

    def __init__(self, normalized_df):
        """
        Initializes the WeightsCalculator class with normalized historical match data.

        Args:
            normalized_df (pd.DataFrame): DataFrame containing normalized historical match data.
        """
        self.normalized_df = normalized_df

    def get_weights(self, method="linear", decay_factor=-0.1):
        """
        Calculates weights for historical match data based on the specified method.

        Args:
            method (str): The method to use for calculating weights. Options are "linear", "season", and "matchday".
            decay_factor (float): The decay factor to use for the exponential decay method.

        Returns:
            pd.DataFrame: A DataFrame containing the historical match data with calculated weights.
        """
        if method == "linear":
            # Calculate linear weights based on the season
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
            # Calculate exponential weights based on the season
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
            # Calculate weights based on the matchday within the season
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
