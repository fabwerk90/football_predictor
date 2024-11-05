import glob
import os

import pandas as pd
import requests


class GetBundesligaData:
    def __init__(self, current_season):
        self.current_season = current_season

    def custom_read_in(self, csv_file):
        df = pd.read_csv(
            csv_file,
            delimiter=",",
            usecols=["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"],
            encoding="ISO-8859–1",
        )
        df["season"] = csv_file.split(".csv")[0][-9:].replace("_", "/")
        return df

    def get_season_fixtures(self):
        season_identifier = (
            f"bundesliga-{self.current_season[0:4]}-WEuropeStandardTime.csv"
        )
        output_file = (
            f"{self.current_season[2:4]}_{self.current_season[7:9]}_fixtures.csv"
        )
        url = f"https://fixturedownload.com/download/{season_identifier}"

        response = requests.get(url)
        filename = f"data/fixtures/{output_file}"
        with open(filename, "wb") as csv_file:
            csv_file.write(response.content)

        return filename

    def get_new_results(self):
        season_identifier = f"{self.current_season[2:4]}{self.current_season[7:9]}"
        output_file = f"{self.current_season[0:4]}_{self.current_season[5:9]}.csv"
        url = f"https://www.football-data.co.uk/mmz4281/{season_identifier}/D1.csv"

        response = requests.get(url)
        if response.status_code != 300:
            filename = f"data/historical_results/{output_file}"
            with open(filename, "wb") as csv_file:
                csv_file.write(response.content)

    def get_historical_results(self, path, seasons_to_keep=5):
        self.get_new_results()
        all_files = glob.glob(os.path.join(path, "*.csv"))
        df_from_each_file = [self.custom_read_in(f) for f in all_files]
        concatenated_df = pd.concat(df_from_each_file, ignore_index=True).dropna()

        current_year = int(self.current_season.split("/")[1])
        last_x_seasons = [
            f"{year}/{year+1}"
            for year in range(current_year - seasons_to_keep, current_year)
        ]
        filtered_df = concatenated_df[concatenated_df["season"].isin(last_x_seasons)]

        return filtered_df

    def team_name_normalization(self, team_names_csv, df):
        team_names = pd.read_csv(team_names_csv, delimiter=";")
        team_names_dict = pd.Series(
            team_names.ger.values, index=team_names.eng
        ).to_dict()

        df["HomeTeam"] = df["HomeTeam"].map(team_names_dict)
        df["AwayTeam"] = df["AwayTeam"].map(team_names_dict)

        normalized_df = df.rename(
            columns={
                "HomeTeam": "home_team",
                "AwayTeam": "away_team",
                "FTHG": "home_goals",
                "FTAG": "away_goals",
                "FTR": "match_outcome",
            }
        )

        return normalized_df

    def get_current_season_average_goals(self, results_df):
        results_df = results_df[results_df["season"] == self.current_season]
        season_goals = results_df.groupby("season")[["home_goals", "away_goals"]].sum()
        n_played_matches = results_df.shape[0]

        season_avg_home_goals = season_goals["home_goals"] / n_played_matches
        season_avg_away_goals = season_goals["away_goals"] / n_played_matches

        return {
            season: {"home": avg_home_goals, "away": avg_away_goals}
            for season, avg_home_goals, avg_away_goals in zip(
                season_avg_home_goals.index,
                season_avg_home_goals,
                season_avg_away_goals,
            )
        }

    def get_season_schedule(self, fixtures_data_csv):
        fixtures_all_cols = pd.read_csv(fixtures_data_csv, delimiter=",")
        replacements = {
            "VfL Bochum 1848": "VfL Bochum",
            "Sport-Club Freiburg": "SC Freiburg",
            "TSG Hoffenheim": "TSG 1899 Hoffenheim",
        }

        fixtures_all_cols["Home Team"].replace(replacements, inplace=True)
        fixtures_all_cols["Away Team"].replace(replacements, inplace=True)

        fixtures_all_cols.rename(
            columns={
                "Home Team": "home_team",
                "Away Team": "away_team",
                "Round Number": "matchday",
            },
            inplace=True,
        )

        return fixtures_all_cols[["matchday", "home_team", "away_team"]]
