import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt

# step 1: download + prep the data
print("1. Downloading 2022-2024 ATP Data...")
years = ['2022', '2023', '2024']
dfs = [pd.read_csv(f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{y}.csv") for y in years]
df = pd.concat(dfs, ignore_index=True)

# FIX: Filter for Top 100 players to reduce noise and variance BEFORE processing
df = df[(df['winner_rank'] <= 100) & (df['loser_rank'] <= 100)].reset_index(drop=True)

# order by date & match_num for chronological order
df = df.sort_values(['tourney_date', 'match_num']).reset_index(drop=True)

# step 2: creating the advanced metrics
print("2. Engineering Advanced Pro Metrics...")

# FIX: Add a tiny epsilon (1e-6) to denominators to prevent division by zero 
# without dropping dominant performances (e.g., facing 0 break points)
epsilon = 1e-6

# metric #1: serve efficiency (aces - double faults)
df['w_serve_eff'] = df['w_ace'] - df['w_df']
df['l_serve_eff'] = df['l_ace'] - df['l_df']

# metric #2: break point conversion % 
df['w_bp_conv'] = (df['l_bpFaced'] - df['l_bpSaved']) / (df['l_bpFaced'] + epsilon)
df['l_bp_conv'] = (df['w_bpFaced'] - df['w_bpSaved']) / (df['w_bpFaced'] + epsilon)

# metric #3: % serve points won 
df['w_spw_pct'] = (df['w_1stWon'] + df['w_2ndWon']) / (df['w_svpt'] + epsilon)
df['l_spw_pct'] = (df['l_1stWon'] + df['l_2ndWon']) / (df['l_svpt'] + epsilon)

# metric #4: return points won % 
df['w_rpw_pct'] = (df['l_svpt'] - df['l_1stWon'] - df['l_2ndWon']) / (df['l_svpt'] + epsilon)
df['l_rpw_pct'] = (df['w_svpt'] - df['w_1stWon'] - df['w_2ndWon']) / (df['w_svpt'] + epsilon)

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
    # FIX: Calculate the global average for this metric to use as a baseline
    global_avg = history[m].mean()
    # FIX: Replace NaN for rookies with the global average instead of 0
    history[f'avg_{m}'] = history.groupby('player_id')[m].transform(lambda x: x.shift().expanding().mean()).fillna(global_avg)

# put these historical averages back into original df "winners"
df = df.merge(history[['tourney_date', 'match_num', 'player_id', 'avg_serve_eff', 'avg_bp_conv', 'avg_spw_pct', 'avg_rpw_pct']],
              left_on=['tourney_date', 'match_num', 'winner_id'], right_on=['tourney_date', 'match_num', 'player_id'], how='left')
df = df.rename(columns={'avg_serve_eff': 'w_avg_serve_eff', 'avg_bp_conv': 'w_avg_bp_conv', 'avg_spw_pct': 'w_avg_spw_pct', 'avg_rpw_pct': 'w_avg_rpw_pct'}).drop(columns=['player_id'])

# repeat for df "losers"
df = df.merge(history[['tourney_date', 'match_num', 'player_id', 'avg_serve_eff', 'avg_bp_conv', 'avg_spw_pct', 'avg_rpw_pct']],
              left_on=['tourney_date', 'match_num', 'loser_id'], right_on=['tourney_date', 'match_num', 'player_id'], how='left')
df = df.rename(columns={'avg_serve_eff': 'l_avg_serve_eff', 'avg_bp_conv': 'l_avg_bp_conv', 'avg_spw_pct': 'l_avg_spw_pct', 'avg_rpw_pct': 'l_avg_rpw_pct'}).drop(columns=['player_id'])

# step 4: machine learning!!!
print("4. Prepping Data for Machine Learning (The Swap)...")

ml_data = df[['w_avg_serve_eff', 'w_avg_bp_conv', 'w_avg_spw_pct', 'w_avg_rpw_pct', 
              'l_avg_serve_eff', 'l_avg_bp_conv', 'l_avg_spw_pct', 'l_avg_rpw_pct']].dropna()

np.random.seed(42)
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
# step 5: train the Random Forest + extract weights
print("5. Training the Pro Random Forest...")

X = final_df.drop('p1_wins', axis=1)
y = final_df['p1_wins']

# Chronological split to prevent data leakage. 
split_idx = int(len(final_df) * 0.75)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- NEW METRICS IMPLEMENTATION ---
# Get hard predictions (0 or 1) for Accuracy and Classification Report
y_pred = model.predict(X_test)
# Get probability of class 1 (Player 1 winning) for ROC-AUC and Brier Score
y_prob = model.predict_proba(X_test)[:, 1] 

print("\n" + "="*40)
print("🏆 MODEL PERFORMANCE METRICS 🏆")
print("="*40)

# 1. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy:  {accuracy * 100:.2f}%")

# 2. ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score:     {roc_auc:.4f}  <-- (1.0 is perfect, 0.5 is random)")

# 3. Brier Score
brier = brier_score_loss(y_test, y_prob)
print(f"Brier Score:       {brier:.4f}  <-- (Closer to 0.0 is better probability calibration)")

# 4. Detailed Classification Report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['P2 Wins (0)', 'P1 Wins (1)']))
print("="*40 + "\n")
# ----------------------------------

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

w_ranks = df[['winner_name', 'winner_rank', 'tourney_date']].rename(columns={'winner_name': 'name', 'winner_rank': 'rank'})
l_ranks = df[['loser_name', 'loser_rank', 'tourney_date']].rename(columns={'loser_name': 'name', 'loser_rank': 'rank'})
all_ranks = pd.concat([w_ranks, l_ranks]).sort_values('tourney_date', ascending=False)

latest_ranks = all_ranks.drop_duplicates(subset=['name'], keep='first')

top_10 = latest_ranks[latest_ranks['rank'] <= 10].sort_values('rank')
for index, row in top_10.iterrows():
    print(f"#{int(row['rank'])}: {row['name']}")

def get_latest_stats(player_name):
    as_winner = df[df['winner_name'] == player_name].copy()
    as_loser = df[df['loser_name'] == player_name].copy()
    
    if as_winner.empty and as_loser.empty:
        print(f"  -> Error: Could not find '{player_name}'. Check spelling!")
        return None 
        
    last_win_date = as_winner['tourney_date'].max() if not as_winner.empty else 0
    last_loss_date = as_loser['tourney_date'].max() if not as_loser.empty else 0
    
    # FIX: Always sort by both tourney_date AND match_num to grab the true final match of a tournament
    if last_win_date > last_loss_date:
        latest_match = as_winner.sort_values(['tourney_date', 'match_num']).iloc[-1]
        return {'avg_serve_eff': latest_match['w_avg_serve_eff'], 'avg_bp_conv': latest_match['w_avg_bp_conv'],
                'avg_spw_pct': latest_match['w_avg_spw_pct'], 'avg_rpw_pct': latest_match['w_avg_rpw_pct']}
    else:
        latest_match = as_loser.sort_values(['tourney_date', 'match_num']).iloc[-1]
        return {'avg_serve_eff': latest_match['l_avg_serve_eff'], 'avg_bp_conv': latest_match['l_avg_bp_conv'],
                'avg_spw_pct': latest_match['l_avg_spw_pct'], 'avg_rpw_pct': latest_match['l_avg_rpw_pct']}

# step 7: interactive predictor 
while True:
    print("\n" + "="*40)
    
    p1 = input("Enter Player 1 Name (or type 'quit' to exit): ")
    if p1.lower() == 'quit': 
        print("Exiting predictor...")
        break
        
    p2 = input("Enter Player 2 Name: ")
    
    p1_stats = get_latest_stats(p1)
    p2_stats = get_latest_stats(p2)
    
    if p1_stats and p2_stats:
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
        
        prob_A = model.predict_proba(matchup_A)[0] 
        prob_B = model.predict_proba(matchup_B)[0] 
        
        p1_true_prob = (prob_A[1] + prob_B[0]) / 2
        p2_true_prob = 1.0 - p1_true_prob
        
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
                
            bars = axs[i].bar([p1, p2], [val1, val2], color=['#1f77b4', '#ff7f0e'], edgecolor='black')
            axs[i].set_title(f"{title}\n(Model Weight: {weight:.1f}%)", fontsize=11)
            
            for bar in bars:
                height = bar.get_height()
                axs[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=10)
                        
            y_max = max(val1, val2)
            if y_max > 0:
                axs[i].set_ylim(0, y_max * 1.2)

        plt.tight_layout()
        plt.show()