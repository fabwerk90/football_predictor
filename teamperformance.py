import pandas as pd


class TeamPerformance:
    def __init__(self, weighted_df):
        self.weighted_df = weighted_df

    @staticmethod
    def compute_weighted_average(group, column_name, weight_column):
        return (group[column_name] * group[weight_column]).sum() / group[
            weight_column
        ].sum()

    def calculate_performance(self, home=True):
        team_performance = self.weighted_df[
            [
                "season",
                "weight",
                "home_team" if home else "away_team",
                "home_goals",
                "away_goals",
            ]
        ].rename(
            columns={
                "home_team" if home else "away_team": "team",
                "home_goals": "home_scored" if home else "away_conceded",
                "away_goals": "home_conceded" if home else "away_scored",
            }
        )

        team_performance = team_performance.groupby(
            ["season", "weight", "team"], as_index=False
        )[
            [
                "home_scored" if home else "away_scored",
                "home_conceded" if home else "away_conceded",
            ]
        ].mean()

        weighted_average_scored = team_performance.groupby("team").apply(
            self.compute_weighted_average,
            "home_scored" if home else "away_scored",
            "weight",
        )
        weighted_average_conceded = team_performance.groupby("team").apply(
            self.compute_weighted_average,
            "home_conceded" if home else "away_conceded",
            "weight",
        )

        return pd.DataFrame(
            list(
                zip(
                    weighted_average_scored.index,
                    weighted_average_scored.values,
                    weighted_average_conceded.values,
                )
            ),
            columns=["team", "scored_wa", "conceded_wa"],
        )

    def team_performance_normalization(self, home_performance, away_performance):
        overall_performance = pd.merge(home_performance, away_performance, on="team")

        league_home_scored = home_performance["scored_wa"].mean()
        league_home_conceded = home_performance["conceded_wa"].mean()
        league_away_scored = away_performance["scored_wa"].mean()
        league_away_conceded = away_performance["conceded_wa"].mean()

        overall_performance["home_attack"] = (
            overall_performance["scored_wa_x"] / league_home_scored
        )
        overall_performance["home_defense"] = (
            overall_performance["conceded_wa_x"] / league_home_conceded
        )
        overall_performance["away_attack"] = (
            overall_performance["scored_wa_y"] / league_away_scored
        )
        overall_performance["away_defense"] = (
            overall_performance["conceded_wa_y"] / league_away_conceded
        )

        return overall_performance[
            ["team", "home_attack", "home_defense", "away_attack", "away_defense"]
        ]
