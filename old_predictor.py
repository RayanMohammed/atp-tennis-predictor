import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# step 1: download + prep the data (specify what years; combine dfs of diff years)
print("1. Downloading 2022-2024 ATP Data...")
years = ['2022', '2023', '2024']
dfs = [pd.read_csv(f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{y}.csv") for y in years]
df = pd.concat(dfs, ignore_index=True)
# order by date & match_num for chronological order
df = df.sort_values(['tourney_date', 'match_num']).reset_index(drop=True)

# step 2: creating the advanced metrics
print("2. Engineering Advanced Pro Metrics...")

# checks for 0 break points & avoids it
df['l_bpFaced_safe'] = df['l_bpFaced'].replace(0, np.nan)
df['w_bpFaced_safe'] = df['w_bpFaced'].replace(0, np.nan)
df['w_svpt_safe'] = df['w_svpt'].replace(0, np.nan)
df['l_svpt_safe'] = df['l_svpt'].replace(0, np.nan)

# metric #1: serve efficiency (aces - double faults)
df['w_serve_eff'] = df['w_ace'] - df['w_df']
df['l_serve_eff'] = df['l_ace'] - df['l_df']

# metric #2: break point conversion % (break points won / break points faced (opponent))
df['w_bp_conv'] = (df['l_bpFaced'] - df['l_bpSaved']) / df['l_bpFaced_safe']
df['l_bp_conv'] = (df['w_bpFaced'] - df['w_bpSaved']) / df['w_bpFaced_safe']

# metric #3: % serve points won ((1st + 2nd) / total serve points)
df['w_spw_pct'] = (df['w_1stWon'] + df['w_2ndWon']) / df['w_svpt_safe']
df['l_spw_pct'] = (df['l_1stWon'] + df['l_2ndWon']) / df['l_svpt_safe']

# metric #4: % return points won (opponent((total serves - points won) / total serves))
# Metric 4: Return Points Won % ((Opponent Total Serves - Opponent Points Won) / Opponent Total Serves)
df['w_rpw_pct'] = (df['l_svpt'] - df['l_1stWon'] - df['l_2ndWon']) / df['l_svpt_safe']
df['l_rpw_pct'] = (df['w_svpt'] - df['w_1stWon'] - df['w_2ndWon']) / df['w_svpt_safe']

# extract only the winner columns into df "winners"
winners = df[['tourney_date', 'match_num', 'winner_id', 'w_serve_eff', 'w_bp_conv', 'w_spw_pct', 'w_rpw_pct']].rename(
    columns={'winner_id': 'player_id', 'w_serve_eff': 'serve_eff', 'w_bp_conv': 'bp_conv', 'w_spw_pct': 'spw_pct', 'w_rpw_pct': 'rpw_pct'})

# extract only the loser columns into df "losers"
losers = df[['tourney_date', 'match_num', 'loser_id', 'l_serve_eff', 'l_bp_conv', 'l_spw_pct', 'l_rpw_pct']].rename(
    columns={'loser_id': 'player_id', 'l_serve_eff': 'serve_eff', 'l_bp_conv': 'bp_conv', 'l_spw_pct': 'spw_pct', 'l_rpw_pct': 'rpw_pct'})

# combine to create timeline of performances by player
history = pd.concat([winners, losers]).sort_values(['tourney_date', 'match_num'])

# step 3: historical averages
print("3. Calculating Historical Rolling Averages...")
metrics = ['serve_eff', 'bp_conv', 'spw_pct', 'rpw_pct']
for m in metrics:
    history[f'avg_{m}'] = history.groupby('player_id')[m].transform(lambda x: x.shift().expanding().mean()).fillna(0)

# put these historical averages back into original df "winners" without altering the df
df = df.merge(history[['tourney_date', 'match_num', 'player_id', 'avg_serve_eff', 'avg_bp_conv', 'avg_spw_pct', 'avg_rpw_pct']],
              left_on=['tourney_date', 'match_num', 'winner_id'], right_on=['tourney_date', 'match_num', 'player_id'], how='left')
df = df.rename(columns={'avg_serve_eff': 'w_avg_serve_eff', 'avg_bp_conv': 'w_avg_bp_conv', 'avg_spw_pct': 'w_avg_spw_pct', 'avg_rpw_pct': 'w_avg_rpw_pct'}).drop(columns=['player_id'])

# repeat for df "losers"
df = df.merge(history[['tourney_date', 'match_num', 'player_id', 'avg_serve_eff', 'avg_bp_conv', 'avg_spw_pct', 'avg_rpw_pct']],
              left_on=['tourney_date', 'match_num', 'loser_id'], right_on=['tourney_date', 'match_num', 'player_id'], how='left')
df = df.rename(columns={'avg_serve_eff': 'l_avg_serve_eff', 'avg_bp_conv': 'l_avg_bp_conv', 'avg_spw_pct': 'l_avg_spw_pct', 'avg_rpw_pct': 'l_avg_rpw_pct'}).drop(columns=['player_id'])

# step 4: machine learning!!!
print("4. Prepping Data for Machine Learning (The Swap)...")

# isolate the columns we want to be placed in the Random Forest
ml_data = df[['w_avg_serve_eff', 'w_avg_bp_conv', 'w_avg_spw_pct', 'w_avg_rpw_pct', 
              'l_avg_serve_eff', 'l_avg_bp_conv', 'l_avg_spw_pct', 'l_avg_rpw_pct']].dropna()

# set a random seed so its reproducible
np.random.seed(42)
# swap the players if True; keep if False
swap = np.random.rand(len(ml_data)) > 0.5
final_df = pd.DataFrame({
    'p1_avg_serve_eff': np.where(swap, ml_data['l_avg_serve_eff'], ml_data['w_avg_serve_eff']),
    'p2_avg_serve_eff': np.where(swap, ml_data['w_avg_serve_eff'], ml_data['l_avg_serve_eff']),
    
    'p1_avg_bp_conv': np.where(swap, ml_data['l_avg_bp_conv'], ml_data['w_avg_bp_conv']),
    'p2_avg_bp_conv': np.where(swap, ml_data['w_avg_bp_conv'], ml_data['l_avg_bp_conv']),
    
    'p1_avg_spw_pct': np.where(swap, ml_data['l_avg_spw_pct'], ml_data['w_avg_spw_pct']),
    'p2_avg_spw_pct': np.where(swap, ml_data['w_avg_spw_pct'], ml_data['l_avg_spw_pct']),
    
    'p1_avg_rpw_pct': np.where(swap, ml_data['l_avg_rpw_pct'], ml_data['w_avg_rpw_pct']),
    'p2_avg_rpw_pct': np.where(swap, ml_data['w_avg_rpw_pct'], ml_data['l_avg_rpw_pct']),
    
    'p1_wins': np.where(swap, 0, 1)
})

# step 5: train the Random Forest + extract weights
print("5. Training the Pro Random Forest...")

# X is features + remove the correct answers
X = final_df.drop('p1_wins', axis=1)
# y is the target aka answer key
y = final_df['p1_wins']

# slice data into 75% training; 25% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 100 decision trees in the Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
# train the model!
model.fit(X_train, y_train)

# test the model! also compare predictions with real answers for the confidence
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"--> Advanced Model Accuracy: {accuracy * 100:.2f}%\n")

# get the weights by column & combine into a dict
importances = model.feature_importances_
feat_dict = dict(zip(X.columns, importances))
metric_weights = {
    'avg_spw_pct': (feat_dict['p1_avg_spw_pct'] + feat_dict['p2_avg_spw_pct']) * 100,
    'avg_rpw_pct': (feat_dict['p1_avg_rpw_pct'] + feat_dict['p2_avg_rpw_pct']) * 100,
    'avg_serve_eff': (feat_dict['p1_avg_serve_eff'] + feat_dict['p2_avg_serve_eff']) * 100,
    'avg_bp_conv': (feat_dict['p1_avg_bp_conv'] + feat_dict['p2_avg_bp_conv']) * 100
}

# step 6: user ui + leaderboard setup
print("\n--- CURRENT TOP 10 PLAYERS AVAILABLE ---")

# get the winners & losers and combine
w_ranks = df[['winner_name', 'winner_rank', 'tourney_date']].rename(columns={'winner_name': 'name', 'winner_rank': 'rank'})
l_ranks = df[['loser_name', 'loser_rank', 'tourney_date']].rename(columns={'loser_name': 'name', 'loser_rank': 'rank'})
all_ranks = pd.concat([w_ranks, l_ranks]).sort_values('tourney_date', ascending=False)

# drop duplicates to keep first occurrence of a name; this is their current rank
latest_ranks = all_ranks.drop_duplicates(subset=['name'], keep='first')

# sort the top 10 & print them out
top_10 = latest_ranks[latest_ranks['rank'] <= 10].sort_values('rank')
for index, row in top_10.iterrows():
    print(f"#{int(row['rank'])}: {row['name']}")

# this function's purpose is to search  database & pull player's most recent stats
def get_latest_stats(player_name):
    # get all rows relevant to player (wins & losses)
    as_winner = df[df['winner_name'] == player_name].copy()
    as_loser = df[df['loser_name'] == player_name].copy()
    
    # test edge case if name not there
    if as_winner.empty and as_loser.empty:
        print(f"  -> Error: Could not find '{player_name}'. Check spelling!")
        return None 
        
    # get most recent win/loss & pull from that df
    last_win_date = as_winner['tourney_date'].max() if not as_winner.empty else 0
    last_loss_date = as_loser['tourney_date'].max() if not as_loser.empty else 0
    
    
    if last_win_date > last_loss_date:
        latest_match = as_winner.sort_values('tourney_date').iloc[-1]
        return {'avg_serve_eff': latest_match['w_avg_serve_eff'], 'avg_bp_conv': latest_match['w_avg_bp_conv'],
                'avg_spw_pct': latest_match['w_avg_spw_pct'], 'avg_rpw_pct': latest_match['w_avg_rpw_pct']}
    else:
        latest_match = as_loser.sort_values('tourney_date').iloc[-1]
        return {'avg_serve_eff': latest_match['l_avg_serve_eff'], 'avg_bp_conv': latest_match['l_avg_bp_conv'],
                'avg_spw_pct': latest_match['l_avg_spw_pct'], 'avg_rpw_pct': latest_match['l_avg_rpw_pct']}

# step 7: interactive predictor (goes until user says 'quit')
while True:
    print("\n" + "="*40)
    
    p1 = input("Enter Player 1 Name (or type 'quit' to exit): ")
    if p1.lower() == 'quit': 
        print("Exiting predictor...")
        break
        
    p2 = input("Enter Player 2 Name: ")
    
    # search for names; continues only if names are valid inputs
    p1_stats = get_latest_stats(p1)
    p2_stats = get_latest_stats(p2)
    if p1_stats and p2_stats:
        # to avoid order bias; we get both p1 & p2 and then swap to get averages
        matchup_A = pd.DataFrame({
            'p1_avg_serve_eff': [p1_stats['avg_serve_eff']], 'p2_avg_serve_eff': [p2_stats['avg_serve_eff']],
            'p1_avg_bp_conv': [p1_stats['avg_bp_conv']], 'p2_avg_bp_conv': [p2_stats['avg_bp_conv']],
            'p1_avg_spw_pct': [p1_stats['avg_spw_pct']], 'p2_avg_spw_pct': [p2_stats['avg_spw_pct']],
            'p1_avg_rpw_pct': [p1_stats['avg_rpw_pct']], 'p2_avg_rpw_pct': [p2_stats['avg_rpw_pct']]
        })
        
        matchup_B = pd.DataFrame({
            'p1_avg_serve_eff': [p2_stats['avg_serve_eff']], 'p2_avg_serve_eff': [p1_stats['avg_serve_eff']],
            'p1_avg_bp_conv': [p2_stats['avg_bp_conv']], 'p2_avg_bp_conv': [p1_stats['avg_bp_conv']],
            'p1_avg_spw_pct': [p2_stats['avg_spw_pct']], 'p2_avg_spw_pct': [p1_stats['avg_spw_pct']],
            'p1_avg_rpw_pct': [p2_stats['avg_rpw_pct']], 'p2_avg_rpw_pct': [p1_stats['avg_rpw_pct']]
        })
        

        # .predict_proba() returns the voting split of the 100 trees 
        prob_A = model.predict_proba(matchup_A)[0] 
        prob_B = model.predict_proba(matchup_B)[0] 
        
        # average probabilities from scenario A and B
        p1_true_prob = (prob_A[1] + prob_B[0]) / 2
        p2_true_prob = 1.0 - p1_true_prob
        
        # check if P1 probability > 0.50; win, else P2 wins
        if p1_true_prob > 0.5:
            w_name, l_name, w_stats, l_stats, confidence = p1, p2, p1_stats, p2_stats, p1_true_prob * 100
        else:
            w_name, l_name, w_stats, l_stats, confidence = p2, p1, p2_stats, p1_stats, p2_true_prob * 100
            
        print("\n🎾 MATCH PREDICTION 🎾")
        print(f"Predicted Winner: {w_name} (True Confidence: {confidence:.1f}%)")
        print("-" * 40)
        print(f"Why {w_name} has the edge over {l_name}:")
        
        metrics_map = {
            'avg_spw_pct': 'Serve Points Won',
            'avg_rpw_pct': 'Return Points Won',
            'avg_serve_eff': 'Serve Efficiency (Aces-DFs)',
            'avg_bp_conv': 'Break Point Conversion'
        }
        
        # text summary & formatting
        for key, label in metrics_map.items():
            w_val, l_val = w_stats[key], l_stats[key]
            weight = metric_weights[key] 
            
            if 'pct' in key or 'conv' in key:
                w_str, l_str = f"{w_val*100:.1f}%", f"{l_val*100:.1f}%"
            else:
                w_str, l_str = f"{w_val:.2f}", f"{l_val:.2f}"
                
            if w_val > l_val:
                print(f"  ✅ ADVANTAGE: {label} [Weight: {weight:.1f}%] ({w_str} vs {l_str})")
            elif w_val < l_val:
                print(f"  ❌ WEAKNESS:  {label} [Weight: {weight:.1f}%] ({w_str} vs {l_str})")
            else:
                print(f"  ➖ TIE:       {label} [Weight: {weight:.1f}%] ({w_str} vs {l_str})")
                
        # step 8: graphs for the matchup 
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f"{p1} vs {p2} - Head-to-Head Stats", fontsize=16, fontweight='bold')
        axs = axs.flatten()
        
        for i, (key, title) in enumerate(metrics_map.items()):
            val1 = p1_stats[key]
            val2 = p2_stats[key]
            weight = metric_weights[key]
            
            if 'pct' in key or 'conv' in key:
                val1, val2 = val1 * 100, val2 * 100
                
            # draw the bars & title & chart
            bars = axs[i].bar([p1, p2], [val1, val2], color=['#1f77b4', '#ff7f0e'], edgecolor='black')
            axs[i].set_title(f"{title}\n(Model Weight: {weight:.1f}%)", fontsize=11)
            
            for bar in bars:
                height = bar.get_height()
                axs[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=10)
                        
            # set y axis height so the graph fits
            y_max = max(val1, val2)
            if y_max > 0:
                axs[i].set_ylim(0, y_max * 1.2)

        # some cleaning to keep the graphs easy to look at
        plt.tight_layout()
        # pauses code until graph window is closed
        plt.show()

# boom we done
