import pandas as pd
import numpy as np

def add_features(df):
    """
    Add engineered features to the FPL dataset for machine learning.
    
    Parameters:
    - df (pd.DataFrame): DataFrame with columns including 'season', 'GW', 'element', 'team',
                         'opponent_team', 'was_home', 'position', 'total_points', 'goals_scored',
                         'assists', 'minutes', 'team_h_score', 'team_a_score', etc.
    
    Returns:
    - pd.DataFrame: DataFrame with additional engineered features.
    """
    # Ensure DataFrame is a copy to avoid modifying the original
    df = df.copy()
    
    # Sort for time-based calculations
    df = df.sort_values(['season', 'element', 'GW']).reset_index(drop=True)
    
    # --- Team-Level Features ---
    # Aggregate team-level data per gameweek
    team_df = df.groupby(['season', 'GW', 'team']).agg({
        'opponent_team': 'first',
        'was_home': 'first',
        'team_h_score': 'first',
        'team_a_score': 'first'
    }).reset_index()
    
    # Calculate team_score and opponent_score based on home/away status
    team_df['team_score'] = np.where(team_df['was_home'], 
                                     team_df['team_h_score'], 
                                     team_df['team_a_score'])
    team_df['opponent_score'] = np.where(team_df['was_home'], 
                                         team_df['team_a_score'], 
                                         team_df['team_h_score'])
    
    # Calculate team_points (3 for win, 1 for draw, 0 for loss)
    team_df['team_points'] = np.select(
        [team_df['team_score'] > team_df['opponent_score'],
         team_df['team_score'] == team_df['opponent_score']],
        [3, 1],
        default=0
    )
    
    # Sort for rolling calculations
    team_df = team_df.sort_values(['season', 'team', 'GW'])
    
    # Create team name to ID mapping (assuming 'team' is name and 'team_id' is ID in df)
    team_mapping = df[['team', 'team_id']].drop_duplicates().set_index('team')['team_id'].to_dict()
    
    # Rolling team metrics (shifted to use past data only)
    team_df['team_goals_scored_per_game'] = (team_df.groupby(['season', 'team'])['team_score']
                                            .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()))
    team_df['team_goals_conceded_per_game'] = (team_df.groupby(['season', 'team'])['opponent_score']
                                              .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()))
    team_df['team_win_streak'] = (team_df.groupby(['season', 'team'])['team_points']
                                 .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).sum()))
    
    # Ensure team_df has unique ['season', 'GW', 'team'] combinations
    team_df = team_df.drop_duplicates(subset=['season', 'GW', 'team'])
    
    # Merge team features back to player DataFrame
    df = df.merge(team_df[['season', 'GW', 'team', 'team_points', 'team_goals_scored_per_game', 
                           'team_goals_conceded_per_game', 'team_win_streak']],
                  on=['season', 'GW', 'team'], how='left')
    
    # --- Opponent-Level Features ---
    opponent_df = team_df[['season', 'GW', 'team', 'team_goals_conceded_per_game']].rename(
        columns={'team': 'opponent_team', 'team_goals_conceded_per_game': 'opponent_xgc_per_game'})

    # Add opponent_team_id using the mapping
    opponent_df['opponent_team_id'] = opponent_df['opponent_team'].map(team_mapping)

    # Check for unmapped teams
    if opponent_df['opponent_team_id'].isnull().any():
        print("Warning: Some opponent teams could not be mapped to IDs.")

    # Ensure opponent_df has unique ['season', 'GW', 'opponent_team_id'] combinations
    opponent_df = opponent_df.drop_duplicates(subset=['season', 'GW', 'opponent_team_id'])

    # Merge opponent features using team IDs
    df = df.merge(opponent_df[['season', 'GW', 'opponent_team_id', 'opponent_xgc_per_game']],
                  left_on=['season', 'GW', 'opponent_team'],
                  right_on=['season', 'GW', 'opponent_team_id'],
                  how='left')

    # Drop the redundant 'opponent_team_id' column after merge
    df = df.drop(columns=['opponent_team_id'], errors='ignore')
    
    # --- Player-Level Features ---
    # Exponentially Weighted Moving Average (EWMA) for player points
    df['player_points_ewma'] = (df.groupby(['season', 'element'])['total_points']
                               .transform(lambda x: x.shift(1).ewm(span=5, adjust=False).mean()))
    
    # --- Advanced Features ---
    # Fixture difficulty (using opponent's goals conceded as proxy for xGC)
    df['fixture_difficulty'] = df['opponent_xgc_per_game'].fillna(df['opponent_xgc_per_game'].mean())
    
    # Home/away indicator
    df['is_home'] = df['was_home'].astype(int)
    
    # Clean sheet probability (simplified using team goals conceded)
    df['clean_sheet_prob'] = 1 / (1 + df['team_goals_conceded_per_game']).replace(np.inf, 1)
    
    # Goal involvement rate
    df['goal_involvement_rate'] = ((df['goals_scored'] + df['assists']) / df['minutes']).replace(np.inf, 0).fillna(0)
    
    # Form vs opponent
    df['form_vs_opponent'] = df['player_points_ewma'] / (df['opponent_xgc_per_game'] + 1)
    
    # --- Time-Based Features ---
    # Player's season average points
    df['player_points_season_avg'] = (df.groupby(['season', 'element'])['total_points']
                                     .transform(lambda x: x.shift(1).expanding().mean()))
    df['form_trend'] = df['player_points_ewma'] - df['player_points_season_avg']
    
    # --- Differential Features ---
    df['points_vs_avg'] = df['total_points'] - df['player_points_season_avg']
    
    # --- Categorical Encoding ---
    # One-hot encode position
    position_dummies = pd.get_dummies(df['position'], prefix='pos')
    df = pd.concat([df, position_dummies], axis=1)
    
    # --- Feature Scaling ---
    # Columns to scale (example list, adjust as needed)
    scale_cols = ['team_goals_scored_per_game', 'team_goals_conceded_per_game', 'team_win_streak',
                  'opponent_xgc_per_game', 'player_points_ewma', 'fixture_difficulty', 
                  'goal_involvement_rate', 'form_vs_opponent', 'form_trend', 'points_vs_avg']
    for col in scale_cols:
        if col in df.columns:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    
    # Fill any remaining NaNs with 0 or appropriate defaults
    df = df.fillna(0)
    
    return df

# Load the data
df = pd.read_csv('data/preprocessed_historical_gw_zeroed.csv', low_memory=False)

# Define key columns for identifying duplicates
key_columns = ['season', 'GW', 'element']

# Check for duplicates in the original data
duplicates_original = df.duplicated(subset=key_columns)
print("Duplicates in original data:", duplicates_original.sum())

# Remove duplicates based on key columns
df = df.drop_duplicates(subset=key_columns)
print(f"Removed {duplicates_original.sum()} duplicate rows based on {key_columns}")

# Add the engineered features
df_with_features = add_features(df)

# Check for duplicates after feature engineering
duplicates_after = df_with_features.duplicated(subset=key_columns)
print("Duplicates after feature engineering:", duplicates_after.sum())

# Save the DataFrame to a CSV file
output_file = 'data/feature_engineered_data.csv'
df_with_features.to_csv(output_file, index=False)
print(f"Feature engineered data saved to {output_file}")

# Inspect the result
print(df_with_features.head())
print("New columns added:", set(df_with_features.columns) - set(df.columns))