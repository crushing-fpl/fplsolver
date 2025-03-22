import requests
import pandas as pd
import time

# Load player IDs
players = pd.read_csv('data/fpl_players_raw.csv')
player_ids = players['id'].tolist()

# Scrape history data
history_data = []
for player_id in player_ids:
    url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
    response = requests.get(url)
    data = response.json()
    history = data.get('history', [])
    for game in history:
        game['player_id'] = player_id
        history_data.append(game)
    time.sleep(1)  # Avoid overwhelming the API

# Save to CSV
history_df = pd.DataFrame(history_data)
history_df.to_csv('data/fpl_player_history.csv', index=False)
print(history_df.head())