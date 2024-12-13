import os

from config import CURRENT_SEASON, SEASONS_TO_KEEP
from getbundesligadata import GetBundesligaData
from matchpredictor import MatchPredictor
from teamperformance import TeamPerformance
from weightscalculator import WeightsCalculator

MATCHDAY = 14

# Initialize the class with the current season
bundesliga_data = GetBundesligaData(CURRENT_SEASON)

# Get historical results
results_folder_path = os.path.join(
    os.path.dirname(__file__), "data/historical_results/"
)
historical_df = bundesliga_data.get_historical_results(
    results_folder_path, SEASONS_TO_KEEP
)

# Normalize team names
teamnames_folder_path = os.path.join(
    os.path.dirname(__file__), "data/team_translations/team_names_translation.csv"
)
normalized_df = bundesliga_data.team_name_normalization(
    teamnames_folder_path, historical_df
)

# Get current season average goals
current_season_average_goals = bundesliga_data.get_current_season_average_goals(
    normalized_df
)

# Find the maximum goals scored by one team since 1993
max_goals_scored_by_one_team_since_1993 = max(
    normalized_df["home_goals"].max(), normalized_df["away_goals"].max()
)

# Get season fixtures
season_fixtures_csv = bundesliga_data.get_season_fixtures()

# Get season schedule
season_schedule = bundesliga_data.get_season_schedule(season_fixtures_csv)

##############

weights_calculator = WeightsCalculator(normalized_df)
weighted_df = weights_calculator.get_weights("matchday", -0.025)


#############

team_perf = TeamPerformance(normalized_df)
home_performance = team_perf.calculate_performance(home=True)
away_performance = team_perf.calculate_performance(home=False)
team_performance_normalized = team_perf.team_performance_normalization(
    home_performance, away_performance
)


#############

# Initialize the MatchPredictor class
match_predictor = MatchPredictor(
    team_performance_normalized,
    current_season_average_goals,
    max_goals_scored_by_one_team_since_1993,
)

# Predict matches for a specific matchday
predicted_matchday_results = match_predictor.predict_matches(
    season_schedule, matchday=MATCHDAY
)

print(predicted_matchday_results)
