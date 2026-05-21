# ATP Tennis Match Predictor 🎾

A machine learning project that predicts the outcomes of professional ATP tennis matches using a Random Forest Classifier. The model is trained on match data from 2022–2024 and features a custom data pipeline to prevent data leakage and calculate surface-specific performance metrics.

## Features

* **Real-time Data Fetching:** Automatically downloads and parses the latest match data directly from the Jeff Sackmann ATP dataset.
* **Surface-Specific Intelligence:** Calculates rolling historical averages for players distinctly across Hard, Clay, and Grass courts.
* **The "Swap" Technique:** Randomizes player positions (P1 vs P2) during training to prevent the model from blindly learning column biases (predicting the left column to always win).
* **Leak-Proof Architecture:** Uses `.expanding().mean()` and precise `tourney_id` matching to ensure the model never uses future data to predict past matches.
* **Interactive CLI Predictor:** A terminal-based UI allowing users to input any two players and a surface to see a simulated prediction with a confidence score.
* **Visual Diagnostics:** Generates Matplotlib-based visualizations including a Confusion Matrix and a Feature Importance Map to prove model validity.

## Engineered Metrics

The model goes beyond standard stats to calculate elite performance indicators:
* **Serve Efficiency:** Ace-to-Double Fault ratio.
* **Break Point Conversion:** Success rate in converting return opportunities.
* **Service/Return Points Won %:** The foundational metrics of professional tennis.
* **Contextual Features:** ATP Rank and Player Age.

## Tech Stack

* **Python 3**
* **scikit-learn** (RandomForestClassifier, Evaluation Metrics)
* **Pandas & NumPy** (Data manipulation, Rolling Window Calculations)
* **Matplotlib** (Visualizing Confusion Matrices & Feature Weights)
* **Requests** (Secure data pipeline)

## How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/RayanMohammed/atp-tennis-predictor.git
cd atp-tennis-predictor

```

2. Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib requests

```

3. Run the script:

```bash
python3 new_predictor.py

```

## Model Evaluation

The model uses strict Overfitting Checks, capping tree depth (`max_depth=8`) and ensuring minimum leaf samples.

* **Overall Accuracy:** ~62-65% (World-class baseline for volatile sports predictions)
* **Bias Check:** The confusion matrix routinely proves a balanced error rate, proving the target-swap technique successfully eliminated spatial bias.
* **Primary Drivers:** ATP Rank and Return Points Won % consistently register as the heaviest weighted features, aligning with real-world tennis logic.

```

```