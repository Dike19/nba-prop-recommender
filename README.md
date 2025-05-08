# nba-prop-recommender

# Basketball Prediction

## Introduction
Welcome to the Basketball Prediction Dashboard, a Flask-based web application designed to predict the likelihood of NBA players exceeding specific statistical thresholds (Points, Rebounds, Assists) in upcoming games. Powered by machine learning models (CatBoost), this app leverages historical data from the 2024-25 season to provide insightful predictions. Whether you're a fan, analyst, or bettor, this tool helps you make informed decisions based on player performance trends.

### Features
- **Player Predictions**: Select a player, team, and opponent to get probability predictions for Points, Rebounds, and Assists, with customizable thresholds.
- **Leaderboard**: View the top 10 players by average Points, Rebounds, and Assists.
- **Player Comparison**: Compare the stats of two players side by side.
- **User Authentication**: Secure access with a simple login system.
- **Responsive Design**: Enjoy a clean, grid-based interface optimized for desktop and mobile.

### Requirements
- Python 3.x
- Required packages: `flask`, `catboost`, `pandas`, `joblib`, `numpy`, `scipy`
- Historical data file: `data/database_24_25.csv`
- Pre-trained models: `models/points_model.cbm`, `models/rebounds_model.cbm`, `models/assists_model.cbm`, and corresponding encoders (`_le_team.pkl`)

### Installation
1. Clone the repository or download the files.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt