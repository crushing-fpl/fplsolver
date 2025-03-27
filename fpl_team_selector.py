import pandas as pd

# Load GW30–GW38 predictions
future_predictions = pd.read_csv('data/gw30_gw38_predictions.csv')

# Aggregate predictions by averaging predicted points across GW30–GW38
aggregated_predictions = future_predictions.groupby('element').agg({
    'predicted_points': 'mean',
    'name': 'first',
    'position': 'first',
    'team': 'first',
    'value': 'first'
}).reset_index()

# Check required columns
required_cols = ['element', 'name', 'position', 'team', 'value', 'predicted_points']
missing_cols = [col for col in required_cols if col not in aggregated_predictions.columns]
if missing_cols:
    raise ValueError(f"Missing columns in aggregated data: {missing_cols}")

# Filter out players with missing positions and map 'AM' to 'MID'
aggregated_predictions = aggregated_predictions.dropna(subset=['position'])
aggregated_predictions['position'] = aggregated_predictions['position'].replace('AM', 'MID')

# Define FPL constraints
BUDGET = 100.0  # Total budget in millions
MAX_PER_TEAM = 3  # Max players from one team
POSITION_LIMITS = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
TEAM_SIZE = 15

def min_cost_to_fill(df, position_count, position_limits):
    min_cost = 0.0
    for pos, limit in position_limits.items():
        needed = limit - position_count[pos]
        if needed > 0:
            cheapest = df[df['position'] == pos]['value'].min() / 10.0
            min_cost += cheapest * needed
    return min_cost

# Helper function to check constraints
def meets_constraints(player, total_cost, budget, team_count, max_per_team, selected_elements):
    team = player['team']
    cost = player['value'] / 10.0  # Convert value to millions
    element = player['element']
    return (total_cost + cost <= budget and
            team_count.get(team, 0) < max_per_team and
            element not in selected_elements)

# Helper function to add a player
def add_player(player, selected, total_cost, team_count, position_count, selected_elements):
    team = player['team']
    pos = player['position']
    cost = player['value'] / 10.0
    element = player['element']
    selected.append(player)
    total_cost += cost
    team_count[team] = team_count.get(team, 0) + 1
    position_count[pos] += 1
    selected_elements.add(element)
    return total_cost

# Helper function to add the cheapest player
def add_cheapest_player(df, selected, total_cost, budget, team_count, position_count, selected_elements, position_limits):
    for pos in position_limits.keys():
        if position_count[pos] < position_limits[pos]:
            pos_players = df[(df['position'] == pos) & (~df['element'].isin(selected_elements))].sort_values('value')
            for _, player in pos_players.iterrows():
                if meets_constraints(player, total_cost, budget, team_count, MAX_PER_TEAM, selected_elements):
                    total_cost = add_player(player, selected, total_cost, team_count, position_count, selected_elements)
                    return total_cost
    return total_cost

# Function to select team
def select_team(df, budget, max_per_team, position_limits):
    df = df.copy()
    df['value_metric'] = df['predicted_points'] / (df['value'] / 10.0)
    
    selected = []
    total_cost = 0.0
    team_count = {}
    position_count = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
    selected_elements = set()
    
    # Reserve £4M for second GK
    working_budget = budget #- 4.0
    
    # Phase 1: Minimum requirements
    min_positions = {'GK': 1, 'DEF': 3, 'MID': 2, 'FWD': 1}
    for pos, count in min_positions.items():
        pos_players = df[df['position'] == pos].sort_values('value_metric', ascending=False)
        for _, player in pos_players.iterrows():
            if position_count[pos] < count and meets_constraints(player, total_cost, working_budget, team_count, max_per_team, selected_elements):
                total_cost = add_player(player, selected, total_cost, team_count, position_count, selected_elements)
    
    # Phase 2: Fill to 14 players
    prev_len = len(selected)
    while len(selected) < 14:
        candidates = df[~df['element'].isin(selected_elements)].sort_values('predicted_points', ascending=False)
        player_added = False
        for _, player in candidates.iterrows():
            cost = player['value'] / 10.0
            if (total_cost + cost <= working_budget and
                position_count[player['position']] < position_limits[player['position']] and
                meets_constraints(player, total_cost, working_budget, team_count, max_per_team, selected_elements)):
                # Check budget sufficiency
                temp_total_cost = total_cost + cost
                temp_position_count = position_count.copy()
                temp_position_count[player['position']] += 1
                min_cost_remaining = min_cost_to_fill(df, temp_position_count, position_limits)
                if working_budget - temp_total_cost >= min_cost_remaining:
                    total_cost = add_player(player, selected, total_cost, team_count, position_count, selected_elements)
                    player_added = True
                    break
        if not player_added:
            total_cost = add_cheapest_player(df, selected, total_cost, working_budget, team_count, position_count, selected_elements, position_limits)
            if len(selected) == prev_len:  # No progress made
                print("Cannot add more players due to budget or constraints.")
                break
        prev_len = len(selected)
    
    # Phase 3: Add second GK with remaining budget
    if position_count['GK'] < 2:
        gks = df[(df['position'] == 'GK') & (~df['element'].isin(selected_elements))].sort_values('value')
        for _, player in gks.iterrows():
            cost = player['value'] / 10.0
            if meets_constraints(player, total_cost, budget, team_count, max_per_team, selected_elements):
                total_cost = add_player(player, selected, total_cost, team_count, position_count, selected_elements)
                break
    
    return pd.DataFrame(selected)

# Select the team
team = select_team(aggregated_predictions, BUDGET, MAX_PER_TEAM, POSITION_LIMITS)

# Function to select starting 11
def select_starting_11(team):
    starting_positions = {'GK': 1, 'DEF': 3, 'MID': 2, 'FWD': 1}
    starting = pd.DataFrame()
    for pos, count in starting_positions.items():
        pos_players = team[team['position'] == pos].sort_values('predicted_points', ascending=False).head(count)
        starting = pd.concat([starting, pos_players])
    
    remaining = team[~team['element'].isin(starting['element'])]
    additional = remaining.sort_values('predicted_points', ascending=False).head(11 - len(starting))
    starting_11 = pd.concat([starting, additional])
    return starting_11

# Select starting 11 and bench
starting_11 = select_starting_11(team)
bench = team[~team['element'].isin(starting_11['element'])].sort_values('predicted_points', ascending=False).head(4)

# Display results
print("\nSelected Team for GW30–GW38:")
print(team[['name', 'position', 'team', 'value', 'predicted_points']])
print(f"Total Cost: {sum(team['value']) / 10.0:.1f} million")
print(f"Total Predicted Points: {sum(team['predicted_points']):.2f}")

# Select captain
captain = team.loc[team['predicted_points'].idxmax()]
print(f"Captain: {captain['name']} ({captain['position']}) with {captain['predicted_points']:.2f} points")

print("\nStarting 11:")
print(starting_11[['name', 'position', 'team', 'predicted_points']])
print("\nBench:")
print(bench[['name', 'position', 'team', 'predicted_points']])