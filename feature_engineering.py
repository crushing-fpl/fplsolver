import pandas as pd
import numpy as np

def add_features(df):
    df = df.copy()
    df = df.sort_values(['season', 'element', 'GW']).reset_index(drop=True)

    # --- Team-Level Features ---
    team_df = df.groupby(['season', 'GW', 'team']).agg({
        'opponent_team': 'first',
        'was_home': 'first',
        'team_h_score': 'first',
        'team_a_score': 'first'
    }).reset_index()

    team_df['team_score'] = np.where(team_df['was_home'], team_df['team_h_score'], team_df['team_a_score'])
    team_df['opponent_score'] = np.where(team_df['was_home'], team_df['team_a_score'], team_df['team_h_score'])
    team_df['team_points'] = np.select(
        [team_df['team_score'] > team_df['opponent_score'],
         team_df['team_score'] == team_df['opponent_score']],
        [3, 1],
        default=0
    )
    team_df = team_df.sort_values(['season', 'team', 'GW'])

    team_df['team_goals_scored_last_5'] = team_df.groupby(['season', 'team'])['team_score'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    team_df['team_goals_conceded_last_5'] = team_df.groupby(['season', 'team'])['opponent_score'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    team_df['team_win_streak'] = team_df.groupby(['season', 'team'])['team_points'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )
    # Add team_form_momentum in team_df
    team_df['team_form_momentum'] = (
        team_df.groupby(['season', 'team'])['team_score'].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        ) - team_df['team_goals_scored_last_5']
    )
    team_df = team_df.drop_duplicates(subset=['season', 'GW', 'team'])
    df = df.merge(team_df[['season', 'GW', 'team', 'team_points', 'team_goals_scored_last_5', 
                          'team_goals_conceded_last_5', 'team_win_streak', 'team_form_momentum']],
                  on=['season', 'GW', 'team'], how='left')

    # --- Opponent-Level Features ---
    opponent_df = team_df[['season', 'GW', 'team', 'team_goals_conceded_last_5']].rename(
        columns={'team': 'opponent_team', 'team_goals_conceded_last_5': 'opponent_goals_conceded_last_5'}
    )
    df = df.merge(opponent_df[['season', 'GW', 'opponent_team', 'opponent_goals_conceded_last_5']],
                  on=['season', 'GW', 'opponent_team'], how='left')

    # --- Player-Level Features ---
    df['player_points_ewma'] = df.groupby(['season', 'element'])['total_points'].transform(
        lambda x: x.shift(1).ewm(span=5, adjust=False).mean()
    )
    df['goals_scored_last_3'] = df.groupby('element')['goals_scored'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df['assists_last_3'] = df.groupby('element')['assists_x'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df['minutes_last_3'] = df.groupby('element')['minutes'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df['bps_last_3'] = df.groupby('element')['bps'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    # --- Advanced Features ---
    df['fixture_difficulty'] = df['opponent_goals_conceded_last_5'].fillna(df['opponent_goals_conceded_last_5'].mean())
    df['is_home'] = df['was_home'].astype(int)
    df['clean_sheet_prob'] = np.exp(-df['team_goals_conceded_last_5'].fillna(df['team_goals_conceded_last_5'].mean()))
    df['goal_involvement_rate'] = ((df['goals_scored'] + df['assists_x']) / df['minutes']).replace(np.inf, 0).fillna(0)
    df['form_vs_opponent'] = df.groupby(['element', 'opponent_team'])['total_points'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    if 'xG' in df.columns:
        df['xg_last_3'] = df.groupby('element')['xG'].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )
    if 'xA' in df.columns:
        df['xa_last_3'] = df.groupby('element')['xA'].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )

    # --- Recommended Features ---
    # Opponent Strength: Average goals conceded by opponent over last 5 games
    df['opponent_strength'] = df.groupby(['season', 'opponent_team'])['opponent_goals_conceded_last_5'].transform('mean')
    # Interaction Term: Player's recent goals vs. fixture difficulty
    df['goals_vs_difficulty'] = df['goals_scored_last_3'] * df['fixture_difficulty']

    # --- Performance Boosters ---
    # 1. Player Consistency: Standard deviation of points over last 5 games
    df['player_consistency'] = df.groupby('element')['total_points'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).std()
    )
    # debug
    # print("Columns in df:", df.columns.tolist())
    # print("'minutes' in df:", 'minutes' in df.columns)
    # print("'element' in df:", 'element' in df.columns)
    # print("'team_score' in df:", 'team_score' in df.columns)
    # # 2. Team Form Momentum: Difference in goals scored over last 3 vs. last 5
    # df['team_form_momentum'] = (
    #     df.groupby(['season', 'team'])['team_score'].transform(
    #         lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    #     ) - df['team_goals_scored_last_5']
    # )
    # 3. Recent Bonus Points Trend: Bonus points over last 3 games
    df['bonus_last_3'] = df.groupby('element')['bonus'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    # --- Time-Based Features ---
    df['player_points_season_avg'] = df.groupby(['season', 'element'])['total_points'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df['form_trend'] = df['player_points_ewma'] - df['player_points_season_avg']

    # --- Differential Features ---
    df['points_vs_avg'] = df['total_points'] - df['player_points_season_avg']

    # --- Categorical Encoding ---
    position_dummies = pd.get_dummies(df['position'], prefix='pos')
    df = pd.concat([df, position_dummies], axis=1)
    position_weights = {'FWD': 1.2, 'MID': 1.0, 'DEF': 0.8, 'GK': 0.7}
    df['position_weight'] = df['position'].map(position_weights).fillna(1.0)

    # --- Feature Scaling ---
    scale_cols = ['team_goals_scored_last_5', 'team_goals_conceded_last_5', 'team_win_streak',
                  'opponent_goals_conceded_last_5', 'player_points_ewma', 'fixture_difficulty', 
                  'goal_involvement_rate', 'form_vs_opponent', 'form_trend', 'points_vs_avg',
                  'opponent_strength', 'goals_vs_difficulty', 'player_consistency', 
                  'team_form_momentum', 'bonus_last_3']  # Updated with new features
    for col in scale_cols:
        if col in df.columns:
            df[col] = (df[col] - df[col].mean()) / df[col].std()

    # Fill any remaining NaNs with 0
    df = df.fillna(0)
    
    return df

# Load the data
df = pd.read_csv('data/combined_historical_gw_with_understat.csv', low_memory=False)

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