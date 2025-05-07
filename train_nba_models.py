import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    """
    Load the NBA dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        required_columns = ['Player', 'Tm', 'Opp', 'MP', 'PTS', 'TRB', 'AST', 'Date']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def compute_rolling_averages(df, stat, window=5):
    """
    Compute rolling averages for a stat per player.
    """
    df = df.sort_values(['Player', 'Date'])
    df[f'{stat}_Avg'] = df.groupby('Player')[stat].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    return df

def prepare_data(df, stat):
    """
    Prepare data for model training by encoding categorical variables and selecting features.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        stat (str): Target statistic to predict (e.g., 'PTS').
    
    Returns:
        tuple: (X, y, le_team) - Features, target, and label encoder.
    """
    try:
        if stat not in df.columns:
            raise ValueError(f"Target stat '{stat}' not found in dataset")
        
        df = df.dropna(subset=['Tm', 'Opp', 'MP', stat, 'FG%', '3PA', 'FTA', 'TRB', 'AST'])
        df = df[df['MP'] > 0]
        
        df = compute_rolling_averages(df, stat)
        
        all_teams = pd.concat([df['Tm'], df['Opp']]).unique()
        le_team = LabelEncoder()
        le_team.fit(all_teams)
        
        df['Team_Encoded'] = le_team.transform(df['Tm'])
        df['Opponent_Encoded'] = le_team.transform(df['Opp'])
        
        features = [
            'Team_Encoded', 'Opponent_Encoded', 'MP', 'FG%', '3PA', 'FTA',
            'TRB', 'AST', f'{stat}_Avg'
        ]
        X = df[features]
        y = df[stat]
        
        logger.info(f"Prepared data for '{stat}' with {len(X)} samples and features: {features}")
        return X, y, le_team
    except Exception as e:
        logger.error(f"Error in prepare_data: {e}")
        raise

def train_model(X, y, stat):
    """
    Train a CatBoost model for the given stat.
    
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        stat (str): Name of the target stat.
    
    Returns:
        CatBoostRegressor: Trained model.
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = CatBoostRegressor(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            random_seed=42,
            verbose=0
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Model for '{stat}' - MSE: {mse:.2f}, R2: {r2:.2f}")
        return model
    except Exception as e:
        logger.error(f"Error in train_model: {e}")
        raise

def save_model_and_encoders(model, le_team, stat, output_dir='models'):
    """
    Save the trained model and label encoder.
    
    Args:
        model: Trained model.
        le_team: Label encoder for teams.
        stat (str): Name of the target stat (e.g., 'points', 'rebounds', 'assists').
        output_dir (str): Directory to save models and encoders.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = os.path.join(output_dir, f"{stat}_model.cbm")
        model.save_model(model_path)
        
        encoder_path = os.path.join(output_dir, f"{stat}_le_team.pkl")
        joblib.dump(le_team, encoder_path)
        
        logger.info(f"Saved model and encoder for '{stat}' to {output_dir}")
    except Exception as e:
        logger.error(f"Error saving model/encoders: {e}")
        raise

def main():
    """
    Main function to load data, prepare it, train models, and save results.
    """
    try:
        file_path = 'data/database_24_25.csv'
        target_stats = {
            'PTS': 'points',
            'TRB': 'rebounds',
            'AST': 'assists'
        }
        
        df = load_data(file_path)
        
        for stat, model_name in target_stats.items():
            logger.info(f"Training model for {stat}")
            X, y, le_team = prepare_data(df, stat)
            model = train_model(X, y, stat)
            save_model_and_encoders(model, le_team, model_name)
            
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()