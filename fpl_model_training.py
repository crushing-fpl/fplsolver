import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

# Define the add_features function directly in this script
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
    df['opponent_strength'] = df.groupby(['season', 'opponent_team'])['opponent_goals_conceded_last_5'].transform('mean')
    df['goals_vs_difficulty'] = df['goals_scored_last_3'] * df['fixture_difficulty']

    # --- Performance Boosters ---
    df['player_consistency'] = df.groupby('element')['total_points'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).std()
    )
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

    # Experimental Features
    # Longer History: 10-game averages
    df['goals_scored_last_10'] = df.groupby('element')['goals_scored'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )
    df['assists_last_10'] = df.groupby('element')['assists_x'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )

    # Injury Risk: Flag if minutes in last 3 games are below 60
    df['low_minutes_flag'] = (df['minutes_last_3'] < 60).astype(int)

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
                  'team_form_momentum', 'bonus_last_3']
    for col in scale_cols:
        if col in df.columns:
            df[col] = (df[col] - df[col].mean()) / df[col].std()

    # Fill any remaining NaNs with 0
    df = df.fillna(0)

    # Drop 'modified' if it exists before returning
    df = df.drop('modified', axis=1, errors='ignore')

    return df

# Load data
df = pd.read_csv('data/feature_engineered_data.csv')
df = df.drop('modified', axis=1, errors='ignore')

# Separate historical and current season data
historical_df = df[df['season'] < '2024-25']
current_df = df[df['season'] == '2024-25']

# Sort historical data
historical_df = historical_df.sort_values(['element', 'season', 'GW'])

# Additional features
historical_df['goals_scored_last_game'] = historical_df.groupby('element')['goals_scored'].shift(1)
historical_df['assists_last_game'] = historical_df.groupby('element')['assists_x'].shift(1)
historical_df['clean_sheets_last_game'] = historical_df.groupby('element')['clean_sheets'].shift(1)
historical_df['minutes_avg_last_5'] = historical_df.groupby('element')['minutes'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
historical_df['bps_last_game'] = historical_df.groupby('element')['bps'].shift(1)
historical_df['player_fatigue'] = historical_df.groupby('element')['minutes'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).sum()
)

# Select features
selected_features = [
    'goals_scored_last_3', 'assists_last_3', 'minutes_last_3', 'bps_last_3',
    'team_goals_scored_last_5', 'team_goals_conceded_last_5',
    'opponent_goals_conceded_last_5', 'form_vs_opponent',
    'player_points_ewma', 'player_points_season_avg', 'value',
    'pos_DEF', 'pos_FWD', 'pos_GK', 'pos_MID',
    'fixture_difficulty', 'is_home', 'position_weight',
    'clean_sheet_prob', 'opponent_strength', 'goals_vs_difficulty',
    'goals_scored_last_10', 'assists_last_10', 'low_minutes_flag',
    'player_fatigue'
]
if 'xG' in historical_df.columns:
    selected_features.extend(['xg_last_3', 'xa_last_3'])

# Prepare dataset
historical_df[selected_features] = historical_df[selected_features].fillna(0)
X = historical_df[selected_features]
y = historical_df['total_points']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and tune Random Forest
param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}
rf_random = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2
)
X_train_sample = X_train.sample(frac=0.1, random_state=42)
y_train_sample = y_train.loc[X_train_sample.index]
rf_random.fit(X_train_sample, y_train_sample)
best_rf = rf_random.best_estimator_
best_rf.fit(X_train, y_train)
y_pred_rf_tuned = best_rf.predict(X_test)
print("Tuned Random Forest RMSE:", root_mean_squared_error(y_test, y_pred_rf_tuned))
print("Tuned Random Forest R2:", r2_score(y_test, y_pred_rf_tuned))

# Feature importance
importances = best_rf.feature_importances_
for name, importance in sorted(zip(selected_features, importances), key=lambda x: x[1], reverse=True):
    print(f"{name}: {importance:.4f}")

# Predict current season (GW1–GW29)
current_df = current_df.sort_values(['element', 'season', 'GW'])
current_df['player_fatigue'] = current_df.groupby('element')['minutes'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).sum()
)
current_df[selected_features] = current_df[selected_features].fillna(0)
current_X = current_df[selected_features]
current_X_scaled = scaler.transform(current_X)
current_predictions = best_rf.predict(current_X_scaled)
current_df['predicted_points'] = current_predictions
output_columns = ['element', 'name', 'position', 'team', 'value', 'predicted_points']
current_df[output_columns].to_csv('data/current_season_predictions.csv', index=False)
print("Current season predictions saved to 'data/current_season_predictions.csv'")

# Function to predict future GWs (GW30–GW38)
def predict_future_gws(model, scaler, selected_features, gw_range=range(30, 39)):
    future_fixtures = pd.read_csv('data/future_fixtures.csv')
    players_df = pd.read_csv('data/fpl_players_raw.csv')
    team_mapping = pd.read_csv('data/fpl_teams_raw.csv').set_index('name')['id'].to_dict()
    
    latest_df = pd.read_csv('data/feature_engineered_data.csv')
    latest_df = latest_df[latest_df['season'] == '2024-25']
    
    if latest_df['team'].dtype == 'object':
        latest_df['team'] = latest_df['team'].map(team_mapping)
    latest_df = latest_df.dropna(subset=['team'])
    latest_df['team'] = latest_df['team'].astype('int64')
    
    latest_df['player_fatigue'] = latest_df.groupby('element')['minutes'].transform(
        lambda x: x.rolling(5, min_periods=1).sum()
    )
    
    player_features = latest_df.groupby('element').last()[selected_features].reset_index()
    team_features = latest_df.groupby('team').last()[['team_goals_scored_last_5', 'team_goals_conceded_last_5']].reset_index()
    
    future_dfs = []
    for gw in gw_range:
        gw_fixtures = future_fixtures[future_fixtures['event'] == gw]
        gw_df = pd.DataFrame()
        
        for _, fixture in gw_fixtures.iterrows():
            home_team_id = fixture['team_h']
            away_team_id = fixture['team_a']
            
            home_players = players_df[players_df['team'] == home_team_id].copy()
            home_players['GW'] = gw
            home_players['team'] = home_team_id
            home_players['opponent_team'] = away_team_id
            home_players['was_home'] = True
            gw_df = pd.concat([gw_df, home_players], ignore_index=True)
            
            away_players = players_df[players_df['team'] == away_team_id].copy()
            away_players['GW'] = gw
            away_players['team'] = away_team_id
            away_players['opponent_team'] = home_team_id
            away_players['was_home'] = False
            gw_df = pd.concat([gw_df, away_players], ignore_index=True)
        
        # Add 'name' and 'position' columns
        gw_df['name'] = gw_df['web_name']
        gw_df['position'] = gw_df['element_type'].map({1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'})
        
        gw_df = gw_df.rename(columns={'id': 'element'})
        gw_df = gw_df.merge(player_features, on='element', how='left')
        gw_df = gw_df.merge(team_features, on='team', how='left')
        
        opponent_team_features = team_features.rename(columns={
            'team': 'opponent_team',
            'team_goals_conceded_last_5': 'opponent_goals_conceded_last_5'
        })
        
        gw_df = gw_df.merge(opponent_team_features[['opponent_team', 'opponent_goals_conceded_last_5']],
                            on='opponent_team', how='left', suffixes=('', '_opponent'))
        
        if 'opponent_goals_conceded_last_5_opponent' in gw_df.columns:
            gw_df['opponent_goals_conceded_last_5'] = gw_df['opponent_goals_conceded_last_5_opponent']
            gw_df = gw_df.drop(columns=['opponent_goals_conceded_last_5_opponent'])
        
        gw_df['is_home'] = gw_df['was_home'].astype(int)
        gw_df['fixture_difficulty'] = gw_df['opponent_goals_conceded_last_5']
        
        for feature in selected_features:
            if feature not in gw_df.columns:
                gw_df[feature] = 0
        
        gw_X = gw_df[selected_features].fillna(0)
        gw_X_scaled = scaler.transform(gw_X)
        gw_df['predicted_points'] = model.predict(gw_X_scaled)
        gw_df['GW'] = gw
        
        inverse_team_mapping = {v: k for k, v in team_mapping.items()}
        gw_df['team'] = gw_df['team'].map(inverse_team_mapping)
        output_columns = ['element', 'name', 'position', 'team', 'value', 'predicted_points']
        future_dfs.append(gw_df[output_columns])
    
    return pd.concat(future_dfs, ignore_index=True)

# Predict GW30–GW38
future_predictions = predict_future_gws(best_rf, scaler, selected_features)
future_predictions.to_csv('data/gw30_gw38_predictions.csv', index=False)
print("GW30–GW38 predictions saved to 'data/gw30_gw38_predictions.csv'")

if __name__ == "__main__":
    pass