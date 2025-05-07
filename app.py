from flask import Flask, render_template, request, redirect, url_for, session
from catboost import CatBoostRegressor
import pandas as pd
import joblib
import logging
import os
import numpy as np
from scipy.stats import norm

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Replace with a secure key

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load historical data
try:
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, "data", "database_24_25.csv")
    model_dir = os.path.join(base_path, "models")
    
    logger.info(f"Attempting to load data from: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    required_columns = ['Player', 'Tm', 'Date']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Get unique players and teams
    players = sorted(df['Player'].unique())
    teams = sorted(df['Tm'].unique())
    
    # Map players to their most recent team
    player_teams = df.groupby('Player')['Tm'].last().to_dict()
    
    logger.info(f"Number of unique players: {len(players)}")
    logger.info(f"Number of unique teams: {len(teams)}")
    logger.debug(f"Sample players: {players[:5]}")
    logger.debug(f"Sample teams: {teams[:5]}")
except Exception as e:
    logger.error(f"Failed to load data: {e}")
    df = pd.DataFrame()
    players = []
    teams = []
    player_teams = {}

def load_model_and_encoder(stat):
    try:
        model_path = os.path.join(model_dir, f"{stat}_model.cbm")
        encoder_path = os.path.join(model_dir, f"{stat}_le_team.pkl")
        
        model = CatBoostRegressor()
        model.load_model(model_path)
        encoder = joblib.load(encoder_path)
        
        logger.info(f"Loaded model and encoder for '{stat}'")
        return model, encoder
    except Exception as e:
        logger.error(f"Error loading model/encoder for '{stat}': {e}")
        raise

def get_historical_averages(player, stat, window=5):
    try:
        player_data = df[df['Player'] == player].sort_values('Date', ascending=False)
        if len(player_data) == 0:
            logger.warning(f"No data for player '{player}'")
            return 0.0
        recent_games = player_data[stat].head(window)
        avg = recent_games.mean() if not recent_games.empty else 0.0
        return avg if not np.isnan(avg) else 0.0
    except Exception as e:
        logger.error(f"Error computing average for '{player}', '{stat}': {e}")
        return 0.0

def estimate_probability(predicted_value, threshold, historical_std, default_std=5.0):
    """
    Estimate the probability of exceeding a threshold using a normal distribution assumption.
    
    Args:
        predicted_value (float): Predicted stat value.
        threshold (float): User-selected or minimum default threshold.
        historical_std (float): Standard deviation of historical data for the stat.
        default_std (float): Default standard deviation if historical data is insufficient.
    
    Returns:
        float: Probability percentage (0-100).
    """
    std = historical_std if historical_std > 0 else default_std
    threshold = float(threshold.replace('+', '')) if threshold else 0.0  # Default to 0 if no threshold, but overridden below
    prob = 1 - norm.cdf(threshold, loc=predicted_value, scale=std)
    return round(prob * 100, 2)

@app.route('/')
def home():
    if 'logged_in' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'password':
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    predictions = None
    error = None
    user_inputs = None
    
    if request.method == 'POST':
        try:
            player = request.form['player']
            team = request.form['team']
            opponent = request.form['opponent']
            
            # Get thresholds (use custom if provided, otherwise dropdown; default to minimum values if empty)
            points_threshold = request.form.get('points_custom') or request.form.get('points_threshold', '')
            rebounds_threshold = request.form.get('rebounds_custom') or request.form.get('rebounds_threshold', '')
            assists_threshold = request.form.get('assists_custom') or request.form.get('assists_threshold', '')
            
            user_inputs = {
                'player': player,
                'team': team,
                'opponent': opponent,
                'points_threshold': points_threshold if points_threshold else '10',  # Default to 10
                'rebounds_threshold': rebounds_threshold if rebounds_threshold else '3',  # Default to 3
                'assists_threshold': assists_threshold if assists_threshold else '3'   # Default to 3
            }
            
            stats = {'points': 'PTS', 'rebounds': 'TRB', 'assists': 'AST'}
            predictions = {}
            
            # Load all models
            models = {}
            encoders = {}
            for model_name in stats.keys():
                models[model_name], encoders[model_name] = load_model_and_encoder(model_name)
            
            # Prepare features once
            team_encoded = encoders['points'].transform([team])[0]
            opponent_encoded = encoders['points'].transform([opponent])[0]
            minutes = 30.0  # Default minutes
            historical_features = {
                'FG%': get_historical_averages(player, 'FG%'),
                '3PA': get_historical_averages(player, '3PA'),
                'FTA': get_historical_averages(player, 'FTA'),
                'TRB': get_historical_averages(player, 'TRB'),
                'AST': get_historical_averages(player, 'AST')
            }
            
            # Calculate predictions for all stats
            for model_name, stat in stats.items():
                threshold = user_inputs[f'{model_name}_threshold']
                model = models[model_name]
                encoder = encoders[model_name]
                
                features = {
                    'Team_Encoded': team_encoded,
                    'Opponent_Encoded': opponent_encoded,
                    'MP': minutes,
                    **{k: v for k, v in historical_features.items()},
                    f'{stat}_Avg': get_historical_averages(player, stat)
                }
                
                input_df = pd.DataFrame([features])
                pred = model.predict(input_df)[0]
                
                # Calculate historical standard deviation
                player_data = df[df['Player'] == player]
                historical_std = player_data[stat].std() if not player_data[stat].empty else 0.0
                
                # Estimate probability with minimum threshold if no input
                prob = estimate_probability(max(0, pred), threshold, historical_std)
                predictions[model_name] = prob  # No 50% fallback, use calculated probability
            
            if error:
                predictions = None
        except Exception as e:
            error = str(e)
            logger.error(f"Error in prediction: {e}")
    
    return render_template('dashboard.html', predictions=predictions, error=error, players=players, teams=teams, player_teams=player_teams, user_inputs=user_inputs)

@app.route('/leaderboard')
def leaderboard():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    # Calculate average stats for leaderboard, replacing NaN with 0
    base_leaderboard = df.groupby('Player').agg({
        'PTS': 'mean',
        'TRB': 'mean',
        'AST': 'mean',
        'Tm': 'last'
    }).reset_index().fillna(0)
    
    # Create three sorted leaderboards
    points_leaderboard = base_leaderboard.sort_values(by='PTS', ascending=False).head(10).to_dict('records')
    rebounds_leaderboard = base_leaderboard.sort_values(by='TRB', ascending=False).head(10).to_dict('records')
    assists_leaderboard = base_leaderboard.sort_values(by='AST', ascending=False).head(10).to_dict('records')
    
    logger.debug(f"Points leaderboard: {points_leaderboard}")
    logger.debug(f"Rebounds leaderboard: {rebounds_leaderboard}")
    logger.debug(f"Assists leaderboard: {assists_leaderboard}")
    
    return render_template(
        'leaderboard.html',
        points_leaderboard=points_leaderboard,
        rebounds_leaderboard=rebounds_leaderboard,
        assists_leaderboard=assists_leaderboard
    )

@app.route('/player_comparison', methods=['GET', 'POST'])
def player_comparison():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    comparison_data = None
    if request.method == 'POST':
        player1 = request.form['player1']
        player2 = request.form['player2']
        
        # Helper function to get player stats
        def get_player_stats(player):
            player_df = df[df['Player'] == player].sort_values('Date', ascending=False)
            if player_df.empty:
                return {'PTS': 0, 'TRB': 0, 'AST': 0, 'FG%': 0, '3PA': 0, 'FTA': 0, 'Tm': 'N/A'}
            
            stats = {
                'PTS': player_df['PTS'].mean(),
                'TRB': player_df['TRB'].mean(),
                'AST': player_df['AST'].mean(),
                'FG%': player_df['FG%'].mean(),
                '3PA': player_df['3PA'].mean(),
                'FTA': player_df['FTA'].mean(),
                'Tm': player_df['Tm'].iloc[0]  # Most recent team
            }
            return {k: (v if not pd.isna(v) else 0) for k, v in stats.items()}
        
        player1_data = get_player_stats(player1)
        player1_data['Player'] = player1
        player2_data = get_player_stats(player2)
        player2_data['Player'] = player2
        
        comparison_data = {'player1': player1_data, 'player2': player2_data}
    
    return render_template('player_comparison.html', players=players, comparison_data=comparison_data)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)