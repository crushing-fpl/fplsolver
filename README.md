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