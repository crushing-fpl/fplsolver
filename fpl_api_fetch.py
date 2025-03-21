import requests
import pandas as pd

# Fetch FPL bootstrap-static data
url = "https://fantasy.premierleague.com/api/bootstrap-static/"
response = requests.get(url)
data = response.json()

# Extract players and teams
players_df = pd.DataFrame(data["elements"])
teams_df = pd.DataFrame(data["teams"])

# Save to CSV
players_df.to_csv("data/fpl_players_raw.csv", index=False)
teams_df.to_csv("data/fpl_teams_raw.csv", index=False)
print("FPL data saved!")