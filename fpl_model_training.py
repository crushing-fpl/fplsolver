import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from xgboost import XGBRegressor

# Load your data and drop the 'modified' column if it exists
df = pd.read_csv('data/feature_engineered_data.csv')
df = df.drop('modified', axis=1, errors='ignore')  # Safely ignore if column isn’t present
print("Columns in df:", df.columns.tolist())

# Sort the DataFrame by player, season, and gameweek
df = df.sort_values(['element', 'season', 'GW'])

# Create lagged features
df['goals_scored_last_game'] = df.groupby('element')['goals_scored'].shift(1)
df['assists_last_game'] = df.groupby('element')['assists_x'].shift(1)
df['clean_sheets_last_game'] = df.groupby('element')['clean_sheets'].shift(1)
df['minutes_avg_last_5'] = df.groupby('element')['minutes'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
df['bps_last_game'] = df.groupby('element')['bps'].shift(1)

# Drop rows where lagged features are NaN
df = df.dropna(subset=['goals_scored_last_game', 'assists_last_game', 
                       'clean_sheets_last_game', 'bps_last_game'])

# Select features, including new ones from feature engineering
selected_features = [
    'goals_scored_last_3', 'assists_last_3', 'minutes_last_3', 'bps_last_3',
    'team_goals_scored_last_5', 'team_goals_conceded_last_5',
    'opponent_goals_conceded_last_5', 'form_vs_opponent',
    'player_points_ewma', 'player_points_season_avg', 'value',
    'pos_DEF', 'pos_FWD', 'pos_GK', 'pos_MID',
    'fixture_difficulty', 'is_home', 'position_weight',
    'clean_sheet_prob',
    'opponent_strength', 'goals_vs_difficulty'  # New features
]
if 'xG' in df.columns:
    selected_features.extend(['xg_last_3', 'xa_last_3'])
print("Selected Features:", selected_features)

# Prepare the Dataset
X = df[selected_features]
y = df['total_points']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

# Normalize Features (for Linear Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Initial Models
# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
print("Linear Regression RMSE:", root_mean_squared_error(y_test, y_pred_lr))
print("Linear Regression R2:", r2_score(y_test, y_pred_lr))

# Random Forest (Untuned)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest RMSE:", root_mean_squared_error(y_test, y_pred_rf))
print("Random Forest R2:", r2_score(y_test, y_pred_rf))

# Random Forest Hyperparameter Tuning with RandomizedSearchCV
print("Starting Random Forest hyperparameter tuning...")
param_dist = {
    'n_estimators': [100, 200, 300, 400],      # Wider range of trees
    'max_depth': [5, 10, 15, 20],              # Shallower and deeper trees
    'min_samples_split': [2, 5, 10, 15],       # More split control options
    'min_samples_leaf': [1, 2, 4],             # New: controls leaf size
    'max_features': ['sqrt', 'log2', None]     # New: controls features per split
}

rf_random = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=50,  # Test 50 random combinations
    cv=3,
    scoring='neg_mean_squared_error',  # Focus on minimizing MSE
    n_jobs=-1,
    verbose=2
)
rf_random.fit(X_train, y_train)
print("Best Parameters:", rf_random.best_params_)

# Evaluate the tuned model on the test set
best_rf = rf_random.best_estimator_
y_pred_rf_tuned = best_rf.predict(X_test)
print("Tuned Random Forest RMSE:", root_mean_squared_error(y_test, y_pred_rf_tuned))
print("Tuned Random Forest R2:", r2_score(y_test, y_pred_rf_tuned))

# XGBoost Model
print("Training XGBoost model...")
xgb = XGBRegressor(random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print("XGBoost RMSE:", root_mean_squared_error(y_test, y_pred_xgb))
print("XGBoost R2:", r2_score(y_test, y_pred_xgb))

# Save predictions (including tuned RF and XGBoost predictions)
results = pd.DataFrame({
    'actual_points': y_test,
    'predicted_points_lr': y_pred_lr,
    'predicted_points_rf': y_pred_rf,
    'predicted_points_rf_tuned': y_pred_rf_tuned,
    'predicted_points_xgb': y_pred_xgb
})
results.to_csv('data/prediction_results.csv', index=False)
print("Predictions saved to 'data/prediction_results.csv'")