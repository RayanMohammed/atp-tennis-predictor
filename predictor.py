import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ==========================================
# STEP 1: DOWNLOAD AND PREP THE RAW DATA
# ==========================================
print("1. Downloading 2022-2024 ATP Data...")

# Define the years we want to pull from Jeff Sackmann's GitHub repository
years = ['2022', '2023', '2024']

# List comprehension: Loops through the years, downloads each CSV directly from the raw GitHub URL, 
# and stores all three dataframes in a list called 'dfs'
dfs = [pd.read_csv(f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{y}.csv") for y in years]

# pd.concat takes our list of 3 dataframes and stacks them vertically into one master DataFrame.
# ignore_index=True resets the row numbers so they go from 0 to 9000 smoothly.
df = pd.concat(dfs, ignore_index=True)

# CRITICAL: We sort by 'tourney_date' and 'match_num' to guarantee strictly chronological order.
# reset_index(drop=True) throws away the old, scrambled row numbers and applies a clean sequential index.
df = df.sort_values(['tourney_date', 'match_num']).reset_index(drop=True)

# ==========================================
# STEP 2: ENGINEER ADVANCED PRO METRICS
# ==========================================
print("2. Engineering Advanced Pro Metrics...")

# DIVISION SAFETY: If a player faced 0 break points, dividing by 0 crashes Python.
# We replace 0s with np.nan (Not a Number). Pandas is smart enough to just ignore NaNs during math.
df['l_bpFaced_safe'] = df['l_bpFaced'].replace(0, np.nan)
df['w_bpFaced_safe'] = df['w_bpFaced'].replace(0, np.nan)
df['w_svpt_safe'] = df['w_svpt'].replace(0, np.nan)
df['l_svpt_safe'] = df['l_svpt'].replace(0, np.nan)

# Metric 1: Serve Efficiency (Total Aces minus Total Double Faults)
df['w_serve_eff'] = df['w_ace'] - df['w_df']
df['l_serve_eff'] = df['l_ace'] - df['l_df']

# Metric 2: Break Point Conversion % (Break Points Won divided by Break Points Faced by the Opponent)
# Winner's BP Won = Loser's BP Faced minus Loser's BP Saved.
df['w_bp_conv'] = (df['l_bpFaced'] - df['l_bpSaved']) / df['l_bpFaced_safe']
df['l_bp_conv'] = (df['w_bpFaced'] - df['w_bpSaved']) / df['w_bpFaced_safe']

# Metric 3: Serve Points Won % ((1st Serve Points Won + 2nd Serve Points Won) / Total Serve Points)
df['w_spw_pct'] = (df['w_1stWon'] + df['w_2ndWon']) / df['w_svpt_safe']
df['l_spw_pct'] = (df['l_1stWon'] + df['l_2ndWon']) / df['l_svpt_safe']

# Metric 4: Return Points Won % ((Opponent Total Serves - Opponent Points Won) / Opponent Total Serves)
df['w_rpw_pct'] = (df['l_svpt'] - df['l_1stWon'] - df['l_2ndWon']) / df['l_svpt_safe']
df['l_rpw_pct'] = (df['w_svpt'] - df['w_1stWon'] - df['w_2ndWon']) / df['w_svpt_safe']

# Extract just the winner columns. .rename() changes 'w_serve_eff' to 'serve_eff' so we have a generic name.
winners = df[['tourney_date', 'match_num', 'winner_id', 'w_serve_eff', 'w_bp_conv', 'w_spw_pct', 'w_rpw_pct']].rename(
    columns={'winner_id': 'player_id', 'w_serve_eff': 'serve_eff', 'w_bp_conv': 'bp_conv', 'w_spw_pct': 'spw_pct', 'w_rpw_pct': 'rpw_pct'})

# Extract the loser columns. .rename() changes 'l_serve_eff' to 'serve_eff' so they match the winners exactly.
losers = df[['tourney_date', 'match_num', 'loser_id', 'l_serve_eff', 'l_bp_conv', 'l_spw_pct', 'l_rpw_pct']].rename(
    columns={'loser_id': 'player_id', 'l_serve_eff': 'serve_eff', 'l_bp_conv': 'bp_conv', 'l_spw_pct': 'spw_pct', 'l_rpw_pct': 'rpw_pct'})

# Stack the winners and losers vertically to create one massive timeline of every player's individual performances.
history = pd.concat([winners, losers]).sort_values(['tourney_date', 'match_num'])

# ==========================================
# STEP 3: CALCULATE HISTORICAL ROLLING AVERAGES
# ==========================================
print("3. Calculating Historical Rolling Averages...")

# Loop through our four new metric columns
metrics = ['serve_eff', 'bp_conv', 'spw_pct', 'rpw_pct']
for m in metrics:
    # groupby('player_id'): Groups data by specific player.
    # .shift(): Pushes data down one row so today's stats aren't included in pre-match averages (NO TIME TRAVEL).
    # .expanding().mean(): Calculates a running, cumulative average of all previous rows.
    # .fillna(0): If it's the player's first match, they have no history (NaN), so we replace it with a 0.
    history[f'avg_{m}'] = history.groupby('player_id')[m].transform(lambda x: x.shift().expanding().mean()).fillna(0)

# Stitch the newly calculated historical averages back onto the original dataset for the Winners.
# how='left' ensures we keep all rows from 'df' and just attach the matching data from 'history'.
df = df.merge(history[['tourney_date', 'match_num', 'player_id', 'avg_serve_eff', 'avg_bp_conv', 'avg_spw_pct', 'avg_rpw_pct']],
              left_on=['tourney_date', 'match_num', 'winner_id'], right_on=['tourney_date', 'match_num', 'player_id'], how='left')
# Rename the columns so we specifically know these are the Winner's historical averages. Drop the redundant player_id column.
df = df.rename(columns={'avg_serve_eff': 'w_avg_serve_eff', 'avg_bp_conv': 'w_avg_bp_conv', 'avg_spw_pct': 'w_avg_spw_pct', 'avg_rpw_pct': 'w_avg_rpw_pct'}).drop(columns=['player_id'])

# Repeat the exact same merge process, but this time attach the stats for the Losers.
df = df.merge(history[['tourney_date', 'match_num', 'player_id', 'avg_serve_eff', 'avg_bp_conv', 'avg_spw_pct', 'avg_rpw_pct']],
              left_on=['tourney_date', 'match_num', 'loser_id'], right_on=['tourney_date', 'match_num', 'player_id'], how='left')
# Rename them to identify them as Loser averages.
df = df.rename(columns={'avg_serve_eff': 'l_avg_serve_eff', 'avg_bp_conv': 'l_avg_bp_conv', 'avg_spw_pct': 'l_avg_spw_pct', 'avg_rpw_pct': 'l_avg_rpw_pct'}).drop(columns=['player_id'])

# ==========================================
# STEP 4: PREP FOR MACHINE LEARNING (THE SWAP)
# ==========================================
print("4. Prepping Data for Machine Learning (The Swap)...")

# Isolate just the columns we want the Random Forest to study. .dropna() removes any rows with broken data.
ml_data = df[['w_avg_serve_eff', 'w_avg_bp_conv', 'w_avg_spw_pct', 'w_avg_rpw_pct', 
              'l_avg_serve_eff', 'l_avg_bp_conv', 'l_avg_spw_pct', 'l_avg_rpw_pct']].dropna()

# Set a random seed so the coin flips are reproducible every time we run the script.
np.random.seed(42)
# Create a True/False array. True = Heads (Swap the players). False = Tails (Keep them as is).
swap = np.random.rand(len(ml_data)) > 0.5

# np.where(condition, true_result, false_result) acts like a fast =IF() statement.
# If swap is True, Player 1 gets the Loser's stats. If False, Player 1 gets the Winner's stats.
final_df = pd.DataFrame({
    'p1_avg_serve_eff': np.where(swap, ml_data['l_avg_serve_eff'], ml_data['w_avg_serve_eff']),
    'p2_avg_serve_eff': np.where(swap, ml_data['w_avg_serve_eff'], ml_data['l_avg_serve_eff']),
    
    'p1_avg_bp_conv': np.where(swap, ml_data['l_avg_bp_conv'], ml_data['w_avg_bp_conv']),
    'p2_avg_bp_conv': np.where(swap, ml_data['w_avg_bp_conv'], ml_data['l_avg_bp_conv']),
    
    'p1_avg_spw_pct': np.where(swap, ml_data['l_avg_spw_pct'], ml_data['w_avg_spw_pct']),
    'p2_avg_spw_pct': np.where(swap, ml_data['w_avg_spw_pct'], ml_data['l_avg_spw_pct']),
    
    'p1_avg_rpw_pct': np.where(swap, ml_data['l_avg_rpw_pct'], ml_data['w_avg_rpw_pct']),
    'p2_avg_rpw_pct': np.where(swap, ml_data['w_avg_rpw_pct'], ml_data['l_avg_rpw_pct']),
    
    # Target Variable: 0 if Player 1 lost (because we swapped them), 1 if Player 1 won.
    'p1_wins': np.where(swap, 0, 1)
})

# ==========================================
# STEP 5: TRAIN THE RANDOM FOREST & EXTRACT WEIGHTS
# ==========================================
print("5. Training the Pro Random Forest...")

# X holds the features (our 8 stat columns). We drop 'p1_wins' because the model can't see the answer key.
X = final_df.drop('p1_wins', axis=1)
# y holds the target (the answer key: 1 or 0).
y = final_df['p1_wins']

# Slice the data: 75% for training (studying), 25% for testing (the final exam).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize the model with 100 decision trees.
model = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the model to find the patterns between the stats (X_train) and the outcomes (y_train).
model.fit(X_train, y_train)

# Ask the model to predict the outcomes of the 25% test data.
# Compare its predictions to the real answers (y_test) to get a percentage score.
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"--> Advanced Model Accuracy: {accuracy * 100:.2f}%\n")

# .feature_importances_ pulls the mathematical weight the model gave to each column.
importances = model.feature_importances_
# Combine the column names with their weights into a dictionary using zip()
feat_dict = dict(zip(X.columns, importances))

# Because P1 and P2 share the same stats, we add their importance scores together to get the total "Category Weight".
metric_weights = {
    'avg_spw_pct': (feat_dict['p1_avg_spw_pct'] + feat_dict['p2_avg_spw_pct']) * 100,
    'avg_rpw_pct': (feat_dict['p1_avg_rpw_pct'] + feat_dict['p2_avg_rpw_pct']) * 100,
    'avg_serve_eff': (feat_dict['p1_avg_serve_eff'] + feat_dict['p2_avg_serve_eff']) * 100,
    'avg_bp_conv': (feat_dict['p1_avg_bp_conv'] + feat_dict['p2_avg_bp_conv']) * 100
}

# ==========================================
# STEP 6: TOP 10 LEADERBOARD & SETUP
# ==========================================
print("\n--- CURRENT TOP 10 PLAYERS AVAILABLE ---")

# Pull winners and losers, rename columns so they match, and stack them vertically.
w_ranks = df[['winner_name', 'winner_rank', 'tourney_date']].rename(columns={'winner_name': 'name', 'winner_rank': 'rank'})
l_ranks = df[['loser_name', 'loser_rank', 'tourney_date']].rename(columns={'loser_name': 'name', 'loser_rank': 'rank'})
all_ranks = pd.concat([w_ranks, l_ranks]).sort_values('tourney_date', ascending=False)

# drop_duplicates keeps the FIRST instance of a name it sees. Since we sorted newest-to-oldest, this grabs their current rank.
latest_ranks = all_ranks.drop_duplicates(subset=['name'], keep='first')
# Filter for ranks 1 through 10, and sort them sequentially.
top_10 = latest_ranks[latest_ranks['rank'] <= 10].sort_values('rank')

# .iterrows() lets us loop through a Pandas dataframe row by row to print out the menu.
for index, row in top_10.iterrows():
    print(f"#{int(row['rank'])}: {row['name']}")

# Create a function to search the database and pull a player's most recent stats
def get_latest_stats(player_name):
    # Grab all rows where the player won or lost
    as_winner = df[df['winner_name'] == player_name].copy()
    as_loser = df[df['loser_name'] == player_name].copy()
    
    # If both are empty, the name was spelled wrong or they aren't in the dataset
    if as_winner.empty and as_loser.empty:
        print(f"  -> Error: Could not find '{player_name}'. Check spelling!")
        return None 
        
    # Find the maximum (newest) date for their wins and their losses
    last_win_date = as_winner['tourney_date'].max() if not as_winner.empty else 0
    last_loss_date = as_loser['tourney_date'].max() if not as_loser.empty else 0
    
    # If their most recent match was a win, pull stats from the 'w_' columns using .iloc[-1] (the last row)
    if last_win_date > last_loss_date:
        latest_match = as_winner.sort_values('tourney_date').iloc[-1]
        return {'avg_serve_eff': latest_match['w_avg_serve_eff'], 'avg_bp_conv': latest_match['w_avg_bp_conv'],
                'avg_spw_pct': latest_match['w_avg_spw_pct'], 'avg_rpw_pct': latest_match['w_avg_rpw_pct']}
    # If their most recent match was a loss, pull stats from the 'l_' columns
    else:
        latest_match = as_loser.sort_values('tourney_date').iloc[-1]
        return {'avg_serve_eff': latest_match['l_avg_serve_eff'], 'avg_bp_conv': latest_match['l_avg_bp_conv'],
                'avg_spw_pct': latest_match['l_avg_spw_pct'], 'avg_rpw_pct': latest_match['l_avg_rpw_pct']}

# ==========================================
# STEP 7: THE INTERACTIVE PREDICTOR
# ==========================================
# Start an infinite loop to ask for predictions until the user types 'quit'
while True:
    print("\n" + "="*40)
    
    # input() stops the program and waits for the user to type in the terminal
    p1 = input("Enter Player 1 Name (or type 'quit' to exit): ")
    if p1.lower() == 'quit': 
        print("Exiting predictor...")
        break # Breaks the infinite loop
        
    p2 = input("Enter Player 2 Name: ")
    
    # Run the names through our search function
    p1_stats = get_latest_stats(p1)
    p2_stats = get_latest_stats(p2)
    
    # If both players were successfully found, proceed to predict
    if p1_stats and p2_stats:
        
        # ENSEMBLING: We create two scenarios to eliminate 'Order Bias'.
        # Scenario A: P1 is Player 1, P2 is Player 2
        matchup_A = pd.DataFrame({
            'p1_avg_serve_eff': [p1_stats['avg_serve_eff']], 'p2_avg_serve_eff': [p2_stats['avg_serve_eff']],
            'p1_avg_bp_conv': [p1_stats['avg_bp_conv']], 'p2_avg_bp_conv': [p2_stats['avg_bp_conv']],
            'p1_avg_spw_pct': [p1_stats['avg_spw_pct']], 'p2_avg_spw_pct': [p2_stats['avg_spw_pct']],
            'p1_avg_rpw_pct': [p1_stats['avg_rpw_pct']], 'p2_avg_rpw_pct': [p2_stats['avg_rpw_pct']]
        })
        
        # Scenario B: P2 is loaded into Player 1 slots, and P1 is loaded into Player 2 slots
        matchup_B = pd.DataFrame({
            'p1_avg_serve_eff': [p2_stats['avg_serve_eff']], 'p2_avg_serve_eff': [p1_stats['avg_serve_eff']],
            'p1_avg_bp_conv': [p2_stats['avg_bp_conv']], 'p2_avg_bp_conv': [p1_stats['avg_bp_conv']],
            'p1_avg_spw_pct': [p2_stats['avg_spw_pct']], 'p2_avg_spw_pct': [p1_stats['avg_spw_pct']],
            'p1_avg_rpw_pct': [p2_stats['avg_rpw_pct']], 'p2_avg_rpw_pct': [p1_stats['avg_rpw_pct']]
        })
        
        # .predict_proba() returns the voting split of the 100 trees (e.g. [0.30, 0.70])
        prob_A = model.predict_proba(matchup_A)[0] 
        prob_B = model.predict_proba(matchup_B)[0] 
        
        # Average the probabilities from Scenario A and Scenario B
        p1_true_prob = (prob_A[1] + prob_B[0]) / 2
        p2_true_prob = 1.0 - p1_true_prob
        
        # If P1's true probability is over 0.50, they win. Otherwise, P2 wins.
        if p1_true_prob > 0.5:
            w_name, l_name, w_stats, l_stats, confidence = p1, p2, p1_stats, p2_stats, p1_true_prob * 100
        else:
            w_name, l_name, w_stats, l_stats, confidence = p2, p1, p2_stats, p1_stats, p2_true_prob * 100
            
        print("\n🎾 MATCH PREDICTION 🎾")
        print(f"Predicted Winner: {w_name} (True Confidence: {confidence:.1f}%)")
        print("-" * 40)
        print(f"Why {w_name} has the edge over {l_name}:")
        
        # Dictionary to map variable names to readable printout labels
        metrics_map = {
            'avg_spw_pct': 'Serve Points Won',
            'avg_rpw_pct': 'Return Points Won',
            'avg_serve_eff': 'Serve Efficiency (Aces-DFs)',
            'avg_bp_conv': 'Break Point Conversion'
        }
        
        # Loop through our metrics to print the text summary
        for key, label in metrics_map.items():
            w_val, l_val = w_stats[key], l_stats[key]
            weight = metric_weights[key] # Fetch the mathematical importance score we calculated earlier
            
            # Format numbers (turn .65 into 65.0%)
            if 'pct' in key or 'conv' in key:
                w_str, l_str = f"{w_val*100:.1f}%", f"{l_val*100:.1f}%"
            else:
                w_str, l_str = f"{w_val:.2f}", f"{l_val:.2f}"
                
            # Print ✅ for advantage and ❌ for weakness based on the values
            if w_val > l_val:
                print(f"  ✅ ADVANTAGE: {label} [Weight: {weight:.1f}%] ({w_str} vs {l_str})")
            elif w_val < l_val:
                print(f"  ❌ WEAKNESS:  {label} [Weight: {weight:.1f}%] ({w_str} vs {l_str})")
            else:
                print(f"  ➖ TIE:       {label} [Weight: {weight:.1f}%] ({w_str} vs {l_str})")
                
        # ==========================================
        # STEP 8: VISUALIZE THE MATCHUP (2x2 GRID)
        # ==========================================
        # plt.subplots creates a 2x2 grid framework for our charts
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f"{p1} vs {p2} - Head-to-Head Stats", fontsize=16, fontweight='bold')
        
        # .flatten() turns the 2x2 grid into a simple 1D array so we can easily loop through it
        axs = axs.flatten()
        
        # enumerate() gives us an index 'i' (0,1,2,3) to place each chart in the correct grid slot
        for i, (key, title) in enumerate(metrics_map.items()):
            val1 = p1_stats[key]
            val2 = p2_stats[key]
            weight = metric_weights[key]
            
            # Convert math decimals to percentages for the chart display
            if 'pct' in key or 'conv' in key:
                val1, val2 = val1 * 100, val2 * 100
                
            # .bar() draws the actual bars. We color P1 blue (#1f77b4) and P2 orange (#ff7f0e)
            bars = axs[i].bar([p1, p2], [val1, val2], color=['#1f77b4', '#ff7f0e'], edgecolor='black')
            # Set the title of the individual subplot
            axs[i].set_title(f"{title}\n(Model Weight: {weight:.1f}%)", fontsize=11)
            
            # Loop through the bars to draw the exact number directly on top of them
            for bar in bars:
                height = bar.get_height()
                axs[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=10)
                        
            # Dynamically set the Y-axis height so the numbers on top don't get cut off
            y_max = max(val1, val2)
            if y_max > 0:
                axs[i].set_ylim(0, y_max * 1.2)

        # .tight_layout() auto-adjusts spacing so titles don't overlap
        plt.tight_layout()
        # Pop open the visual window! The code pauses here until you close it.
        plt.show()