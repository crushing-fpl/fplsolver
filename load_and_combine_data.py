import pandas as pd
import glob
import os

# Path to your historical data
historical_path = 'data/historical/'

# Find all merged_gw.csv files across seasons
gw_files = glob.glob(historical_path + '*/gws/merged_gw.csv')

# Load and concatenate into one DataFrame
df_list = []
for file in gw_files:
    print(f"Loading {file}...")
    try:
        # Extract season from the path (e.g., '2024-25')
        season = os.path.basename(os.path.dirname(os.path.dirname(file)))
        # Read the CSV with latin1 encoding to handle special characters
        df = pd.read_csv(file, encoding='latin1')
        # Add the season column to this DataFrame
        df['season'] = season
        df_list.append(df)
    except Exception as e:
        print(f"Error in file {file}: {e}")
        break  # Stop at the problematic file

# If no errors, concatenate and proceed with team ID mapping
if len(df_list) == len(gw_files):
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Function to load team mapping for a given season
    def load_team_mapping(season):
        """
        Loads team name-to-ID mapping from the teams.csv file for a specific season.
        
        Args:
            season (str): Season in format 'YYYY-YY' (e.g., '2020-21')
        
        Returns:
            dict: Mapping of team name to team ID
        """
        url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/teams.csv"
        try:
            teams_df = pd.read_csv(url)
            return teams_df.set_index('name')['id'].to_dict()
        except Exception as e:
            print(f"Error loading teams for season {season}: {e}")
            return {}
    
    # Step 1: Identify unique seasons in the dataset
    unique_seasons = combined_df['season'].unique()
    
    # Step 2: Load team mappings for all unique seasons
    season_mappings = {season: load_team_mapping(season) for season in unique_seasons}
    
    # Step 3: Function to map team ID based on season and team name
    def map_team_id(row):
        """
        Maps a team name to its ID based on the season.
        
        Args:
            row: DataFrame row with 'season' and 'team' columns
        
        Returns:
            int or None: Team ID if found, None if not
        """
        season = row['season']
        team_name = row['team']
        mapping = season_mappings.get(season, {})
        return mapping.get(team_name, None)
    
    # Step 4: Apply the mapping to add team_id column
    combined_df['team_id'] = combined_df.apply(map_team_id, axis=1)
    
    # Step 5: Save the combined data with team_id
    combined_df.to_csv('data/combined_historical_gw.csv', index=False)
    print("Combined data with team_id saved successfully. First few rows:")
    print(combined_df[['team', 'season', 'team_id']].head())
else:
    print("Fix the error in the file above before proceeding.")

# Optional: Verify the results
if 'combined_df' in locals():
    print("\nVerifying gameweeks per season:")
    for season in combined_df['season'].unique():
        season_data = combined_df[combined_df['season'] == season]
        gw_count = season_data['GW'].nunique()
        print(f"Season {season}: {gw_count} gameweeks")
    
    # Check latest gameweek for 2024-25
    ongoing_season = '2024-25'
    if ongoing_season in combined_df['season'].unique():
        latest_gw = combined_df[combined_df['season'] == ongoing_season]['GW'].max()
        print(f"Latest gameweek for {ongoing_season}: GW{latest_gw}")
    else:
        print(f"No data for season {ongoing_season}")