import pandas as pd
import glob
import os

def load_understat_data(season, understat_path):
    """Load and combine Understat data for a given season."""
    understat_files = glob.glob(os.path.join(understat_path, '*.csv'))
    understat_dfs = []
    for file in understat_files:
        player_name_id = os.path.basename(file).replace('.csv', '')
        player_name = '_'.join(player_name_id.split('_')[:-1])  # Extract name, e.g., "Aaron_Cresswell"
        player_df = pd.read_csv(file)
        player_df['player_name'] = player_name
        player_df['season'] = season
        understat_dfs.append(player_df)
    return pd.concat(understat_dfs, ignore_index=True)

def load_season_data(season, historical_path):
    gw_path = os.path.join(historical_path, season, 'gws', 'merged_gw.csv')
    gw_df = pd.read_csv(gw_path, encoding='ISO-8859-1')
    print(f"Season {season} gw_df columns:", gw_df.columns.tolist())
    
    teams_path = os.path.join(historical_path, season, 'teams.csv')
    teams_df = pd.read_csv(teams_path, encoding='ISO-8859-1')
    team_id_to_name = teams_df.set_index('id')['name'].to_dict()
    gw_df['opponent_team'] = gw_df['opponent_team'].map(team_id_to_name)
    
    players_raw_path = os.path.join(historical_path, season, 'players_raw.csv')
    players_raw = pd.read_csv(players_raw_path, encoding='ISO-8859-1')
    players_raw['player_name'] = players_raw['first_name'] + '_' + players_raw['second_name']
    
    understat_path = os.path.join(historical_path, season, 'understat')
    understat_df = load_understat_data(season, understat_path)
    understat_df['date'] = pd.to_datetime(understat_df['date'], format='mixed', errors='coerce')
    understat_df = understat_df.dropna(subset=['date'])
    understat_df['GW'] = understat_df.groupby('season')['date'].rank(method='dense').astype(int)
    
    gw_df['name'] = gw_df['name'].str.replace(' ', '_')  # Align name formats
    merged_df = pd.merge(
        gw_df, 
        understat_df[['player_name', 'GW', 'xG', 'xA', 'goals', 'assists', 'shots', 'key_passes']], 
        left_on=['name', 'GW'], 
        right_on=['player_name', 'GW'], 
        how='left'
    )
    merged_df = merged_df.drop(columns=['player_name'], errors='ignore')
    merged_df[['xG', 'xA']] = merged_df[['xG', 'xA']].fillna(0)
    print(f"Season {season} merged_df columns:", merged_df.columns.tolist())
    return merged_df

def load_and_combine_data(historical_path, seasons):
    """Load and combine data from multiple seasons, starting from 2021-22."""
    combined_df = pd.DataFrame()
    for season in seasons:
        season_df = load_season_data(season, historical_path)
        season_df['season'] = season
        combined_df = pd.concat([combined_df, season_df], ignore_index=True)
    return combined_df

# Usage
historical_path = 'data/historical/'
seasons = ['2021-22', '2022-23', '2023-24', '2024-25']  # Adjust based on available data
combined_df = load_and_combine_data(historical_path, seasons)
combined_df.to_csv('data/combined_historical_gw_with_understat.csv', index=False)
print("Combined data saved with xG and xA.")