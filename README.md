# fplsolver

## **Structured Overall Plan**
This is our roadmap—a quick reference to keep us aligned at every stage.

### **Project Summary**
- **Goal**: Build a machine learning or neural network solver for Fantasy Premier League (FPL) to predict player points, optimize squad selection (11 starters + 4 bench), plan transfers and chip usage, and leverage public tendencies to target a top 10k ranking.
- **Approach**: Use PyTorch for neural networks, with simpler models as baselines. Pull data from the FPL API and Understat (xG, xA). Balance differential picks and template players for strategic edge.
- **Tools**: Jupyter Notebooks (development), Google Colab (GPU access), GitHub (version control), and Python libraries like Pandas, NumPy, scikit-learn, and PuLP.
- **Timeline**: Working prototype by mid-July, finalized solver by end of July.

### **Step Context**
Here’s how the project breaks down into 10 logical steps:
1. **Data Collection**: Gather historical and live data from FPL API and Understat.
2. **Data Preprocessing**: Clean, merge, and structure the data.
3. **Feature Engineering**: Create predictors like form and fixture difficulty.
4. **Exploratory Data Analysis (EDA)**: Uncover patterns and insights.
5. **Model Development**: Build and test models, starting with a feedforward neural network (FNN).
6. **Model Evaluation**: Assess performance and pick the best model.
7. **Squad Selection Optimization**: Optimize a 15-player squad within constraints.
8. **Transfer and Chip Strategy**: Plan transfers and chip usage for max points.
9. **Pipeline Automation**: Automate data updates and predictions.
10. **Testing and Refinement**: Backtest and polish the solver.

### **Key Resources**
- **Libraries**: PyTorch (neural networks), scikit-learn (baselines), Pandas/NumPy (data handling), Matplotlib/Seaborn (visualization), PuLP (optimization).
- **Platforms**: Google Colab (GPU), GitHub (version control), Jupyter Notebooks (development).
- **Data Sources**: FPL API, Understat (xG, xA).

---

## **Detailed Step Plans**
Each step below includes everything we need to tackle it: objectives, tasks, tools, outputs, challenges, and learning resources. We’ll focus on one step per chat to keep things manageable.

### **Step 1: Data Collection**
- **Overview**: Collect comprehensive data for FPL players, teams, and fixtures.
- **Tasks**:
  - Write scripts to fetch data from the FPL API (player stats, teams, fixtures).
  - Scrape xG and xA data from Understat.
  - Gather 3-5 seasons of historical data.
- **Tools**: Python (requests, BeautifulSoup), Jupyter Notebooks.
- **Expected Outputs**: Raw CSV/JSON files for each data source.
- **Potential Challenges**: API rate limits, inconsistent data formats.
- **Learning Resources**:
  - [FPL API Documentation](https://fantasy.premierleague.com/api/bootstrap-static/)
  - [Understat Scraping Tutorial](https://github.com/amosbastian/fpl)

### **Step 2: Data Preprocessing**
- **Overview**: Clean and unify the data for analysis.
- **Tasks**:
  - Handle missing values and outliers.
  - Merge FPL and Understat datasets.
  - Standardize player and team names.
- **Tools**: Pandas, NumPy.
- **Expected Outputs**: A clean, merged DataFrame (CSV).
- **Potential Challenges**: Naming inconsistencies, missing data for new players.
- **Learning Resources**:
  - [Pandas Data Cleaning Guide](https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html)

### **Step 3: Feature Engineering**
- **Overview**: Build features to predict player points.
- **Tasks**:
  - Compute rolling averages for form (e.g., last 5 games).
  - Rate fixture difficulty based on team strength.
  - Create differential metrics (ownership, potential).
- **Tools**: Pandas, NumPy.
- **Expected Outputs**: An enriched DataFrame with features.
- **Potential Challenges**: Overcomplicating features.
- **Learning Resources**:
  - [Feature Engineering for Time Series](https://machinelearningmastery.com/basic-feature-engineering-time-series-data-python/)

### **Step 4: Exploratory Data Analysis (EDA)**
- **Overview**: Analyze data for patterns and insights.
- **Tasks**:
  - Plot feature-point correlations.
  - Compare differentials vs. template players.
  - Spot outliers.
- **Tools**: Matplotlib, Seaborn.
- **Expected Outputs**: Plots and a summary of findings.
- **Potential Challenges**: Misinterpreting noisy data.
- **Learning Resources**:
  - [Seaborn Correlation Heatmap Tutorial](https://seaborn.pydata.org/examples/many_pairwise_correlations.html)

### **Step 5: Model Development**
- **Overview**: Build models to predict points, starting with an FNN.
- **Tasks**:
  - Split data into train/test sets.
  - Create a baseline (e.g., linear regression).
  - Build an FNN in PyTorch.
  - Tune hyperparameters.
- **Tools**: PyTorch, scikit-learn, Google Colab.
- **Expected Outputs**: Trained models and initial metrics.
- **Potential Challenges**: Overfitting with neural networks.
- **Learning Resources**:
  - [PyTorch Neural Network Tutorial](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)

### **Step 6: Model Evaluation**
- **Overview**: Evaluate and select the best model.
- **Tasks**:
  - Compare models (RMSE, MAE).
  - Analyze errors for key players.
  - Pick the top performer.
- **Tools**: scikit-learn, Matplotlib.
- **Expected Outputs**: A chosen model and evaluation report.
- **Potential Challenges**: Balancing accuracy and generalizability.
- **Learning Resources**:
  - [Model Evaluation Techniques](https://scikit-learn.org/stable/modules/model_evaluation.html)

### **Step 7: Squad Selection Optimization**
- **Overview**: Optimize a 15-player squad within budget and rules.
- **Tasks**:
  - Define the problem (maximize points).
  - Solve with linear programming or heuristics.
  - Enforce constraints (e.g., max 3 per team).
- **Tools**: PuLP, custom algorithms.
- **Expected Outputs**: A squad selection function.
- **Potential Challenges**: Managing multiple constraints.
- **Learning Resources**:
  - [PuLP Documentation](https://coin-or.github.io/pulp/)

### **Step 8: Transfer and Chip Strategy**
- **Overview**: Plan transfers and chips for maximum points.
- **Tasks**:
  - Model transfer impacts (free vs. hits).
  - Simulate chip usage (e.g., Wildcard, Bench Boost).
  - Factor in ownership trends.
- **Tools**: Custom Python logic.
- **Expected Outputs**: A strategy module.
- **Potential Challenges**: Short-term vs. long-term tradeoffs.
- **Learning Resources**:
  - [FPL Strategy Guides](https://www.fantasyfootballscout.co.uk/fantasy-football-strategy/)

### **Step 9: Pipeline Automation**
- **Overview**: Automate data updates and predictions.
- **Tasks**:
  - Script weekly data refreshes.
  - Automate model retraining.
  - Output squad/transfer suggestions.
- **Tools**: Python scripts, cron jobs.
- **Expected Outputs**: An automated pipeline.
- **Potential Challenges**: Ensuring reliability.
- **Learning Resources**:
  - [Automating Python Scripts](https://realpython.com/python-automation/)

### **Step 10: Testing and Refinement**
- **Overview**: Backtest and refine the solver.
- **Tasks**:
  - Simulate past seasons.
  - Adjust based on weaknesses.
  - Finalize for live use.
- **Tools**: Python, historical data.
- **Expected Outputs**: A polished solver.
- **Potential Challenges**: Overfitting to history.
- **Learning Resources**:
  - [Backtesting Strategies](https://towardsdatascience.com/backtesting-trading-strategies-with-python-8d79b9e3b753)

---

### notes from finished steps:
## Step 1: Data Sources

### Current Season Data
- **Source**: Official Fantasy Premier League (FPL) API.
- **Files**:  
  - Player data: `data/fpl_players_raw.csv`  
  - Team data: `data/fpl_teams_raw.csv`  
  - Player history (gameweek-by-gameweek): `data/fpl_player_history.csv` (if collected)  
- **Notes**: Data for the 2024-25 season is updated dynamically as new gameweeks are played. As of now, it includes up to Gameweek 28.

### Historical Data
- **Source**: [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League).  
- **Seasons Included**: 2020-21, 2021-22, 2022-23, 2023-24, 2024-25 (ongoing).  
- **Files**:  
  - Per season: `data/historical/<season>/gws/merged_gw.csv`  
  - Combined: `data/combined_historical_gw.csv`  
- **Additional Data**: Includes Understat metrics (e.g., xG, xA) integrated into `merged_gw.csv` files.

### Limitations
- **Missing Columns**: Some metrics (e.g., `expected_assists`, `starts`) are absent in older seasons (2020-21, 2021-22) as they were introduced later by the FPL API.  
- **Ongoing Season**: The 2024-25 season data currently goes up to GW28. GW29 and beyond will be added as the season progresses.

### Updates
- Current-season data will be refreshed after each gameweek (e.g., GW30 starting Saturday).


#### Step 2: Data Preprocessing

The raw FPL historical gameweek data was cleaned and standardized for analysis. Key tasks included:

- **Handled missing values**: Missing data in performance columns (e.g., `minutes`, `goals_scored`, `assists`, `expected_goals`) were filled with 0, reflecting no participation or unavailable metrics in certain gameweeks or seasons.
- **Standardized data types**:
  - Numerical columns (e.g., `minutes`, `goals_scored`, `team_id`) set to integers.
  - Categorical columns (e.g., `position`, `team`, `season`) set to strings.
  - `kickoff_time` converted to datetime format.
- **Created a derived column**: A simplified `total_points` column was calculated for rough estimates (official `total_points` used for precision).
- **Removed duplicates**: Dropped duplicates based on `element` (player ID), `season`, and `GW` (gameweek) for unique records.
- **Mapped team IDs**: Team IDs linked to names per season to handle promotion/relegation changes.
- **Checked for outliers** (on `data/preprocessed_historical_gw_zeroed.csv`):
  - No impossible values found (e.g., negative minutes, goals, assists, or xG).
  - 15,601 statistical outliers in `total_points` identified via IQR method, but these align with valid high-scoring performances (e.g., 4-10 points).
  - 34,043 rows with `minutes > 0` and `total_points == 0` reviewed; these are plausible for players (e.g., defenders or goalkeepers) in games without positive contributions.

The final preprocessed dataset is saved as `data/preprocessed_historical_gw_zeroed.csv`.

##### Notes
- The dataset includes Understat metrics (e.g., `expected_goals`, `expected_assists`), so no additional merging is needed.
- Outliers and suspicious rows are consistent with FPL scoring patterns, requiring no corrections.


## Step 3: Feature Engineering (Completed)

- **Overview**:  
  The `feature_engineering.py` script enhances the preprocessed FPL dataset by adding engineered features to capture player form, team performance, opponent strength, and more for machine learning.

- **Input**:  
  - `data/preprocessed_historical_gw_zeroed.csv`: Preprocessed gameweek data.

- **Output**:  
  - `data/feature_engineered_data.csv`: Original data plus new features, saved in the `data/` folder.

- **Key Features Added**:  
  1. **Player Rolling Metrics**:  
     - `player_points_ewma`: Exponentially weighted moving average of player points.  
  2. **Team Performance Metrics**:  
     - `team_goals_scored_per_game`  
     - `team_goals_conceded_per_game`  
     - `team_win_streak`  
     - `team_points`  
  3. **Opponent Strength Metrics**:  
     - `opponent_xgc_per_game`: Expected goals conceded by the opponent.  
  4. **Fixture Difficulty**:  
     - `fixture_difficulty`: Based on opponent strength.  
  5. **Position Indicators**:  
     - One-hot encoded position flags (e.g., `pos_GK`, `pos_DEF`).  
  6. **Interaction Features**:  
     - `form_vs_opponent`  
     - `goal_involvement_rate`  
  7. **Time-Based Features**:  
     - `player_points_season_avg`  
     - `form_trend`  
  8. **Differential Features**:  
     - `points_vs_avg`  
  9. **Additional Features**:  
     - `is_home`  
     - `clean_sheet_prob`  

- **Usage**:  
  1. Ensure `data/preprocessed_historical_gw_zeroed.csv` exists.  
  2. Run `%run feature_engineering.py` in a Python environment.  
  3. The script checks for duplicates, adds features, and saves the output.  

- **Dependencies**:  
  - `pandas`  
  - `numpy`  

- **Example Output**:  
  ```bash
  Duplicates in original data: 0
  Removed 0 duplicate rows based on ['season', 'GW', 'element']
  Duplicates after feature engineering: 0
  Feature engineered data saved to data/feature_engineered_data.csv