import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import nfl_data_py as nfl

class NFLMatchupDataset(Dataset):
    """Custom Dataset for NFL matchup data"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class NFLMatchupPredictor(nn.Module):
    """CNN model for NFL matchup prediction"""
    def __init__(self, input_features):
        super(NFLMatchupPredictor, self).__init__()
        
        # Define network architecture
        self.network = nn.Sequential(
            # First layer
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            # Second layer
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            
            # Third layer
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            
            # Output layer
            nn.Linear(32, 2)  # 2 outputs for binary classification
        )
        
    def forward(self, x):
        return self.network(x)

def get_team_stats(team, pbp_data):
    """Calculate statistics for a specific team"""
    team_offense = pbp_data[pbp_data['posteam'] == team]
    team_defense = pbp_data[pbp_data['defteam'] == team]
    
    games_played = len(team_offense['game_id'].unique())
    if games_played == 0:
        return None
        
    # Offensive Stats
    offensive_stats = {
        'points_scored': team_offense.groupby('game_id')['posteam_score'].last().mean(),
        'total_yards_per_game': team_offense['yards_gained'].fillna(0).mean(),
        'pass_yards_per_game': team_offense['passing_yards'].fillna(0).mean(),
        'rush_yards_per_game': team_offense['rushing_yards'].fillna(0).mean(),
        'turnovers_per_game': team_offense['turnover'].sum() / games_played,
        'third_down_conv_rate': (
            team_offense[team_offense['third_down_converted'] == 1].shape[0] /
            max(team_offense[team_offense['down'] == 3].shape[0], 1)
        )
    }
    
    # Defensive Stats
    defensive_stats = {
        'points_allowed': team_defense.groupby('game_id')['defteam_score'].last().mean(),
        'def_yards_allowed_per_game': team_defense['yards_gained'].fillna(0).mean(),
        'def_sacks_per_game': team_defense['sack'].sum() / games_played,
        'def_interceptions_per_game': team_defense['interception'].sum() / games_played,
        'def_third_down_stop_rate': 1 - (
            team_defense[team_defense['third_down_converted'] == 1].shape[0] /
            max(team_defense[team_defense['down'] == 3].shape[0], 1)
        )
    }
    
    return {**offensive_stats, **defensive_stats}

def prepare_matchup_data(year):
    """Prepare matchup data for the specified year"""
    print(f"Loading data for {year}...")
    
    # Load schedule and play-by-play data
    schedule = nfl.import_schedules([year])
    pbp_data = nfl.import_pbp_data([year])
    
    matchup_data = []
    
    # Process each game
    for _, game in schedule.iterrows():
        home_team = game['home_team']
        away_team = game['away_team']
        
        # Get stats for both teams
        home_stats = get_team_stats(home_team, pbp_data)
        away_stats = get_team_stats(away_team, pbp_data)
        
        if home_stats is None or away_stats is None:
            continue
        
        # Combine features for the matchup
        features = np.array(list(home_stats.values()) + list(away_stats.values()))
        
        # Create label (1 if home team wins, 0 if away team wins)
        label = 1 if game['home_score'] > game['away_score'] else 0
        
        matchup_data.append({
            'home_team': home_team,
            'away_team': away_team,
            'features': features,
            'label': label,
            'feature_names': list(home_stats.keys()) + list(away_stats.keys())
        })
    
    return pd.DataFrame(matchup_data)

def predict_matchup(model, scaler, home_team, away_team, pbp_data):
    """Predict the outcome of a specific matchup"""
    # Get current stats for both teams
    home_stats = get_team_stats(home_team, pbp_data)
    away_stats = get_team_stats(away_team, pbp_data)
    
    if home_stats is None or away_stats is None:
        return None
    
    # Combine features
    features = np.array(list(home_stats.values()) + list(away_stats.values())).reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Convert to tensor
    features_tensor = torch.FloatTensor(features_scaled)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = outputs.max(1)[1].item()
    
    return {
        'prediction': home_team if prediction == 1 else away_team,
        'home_win_probability': probabilities[0][1].item(),
        'away_win_probability': probabilities[0][0].item()
    }

def main():
    # Prepare training data
    train_years = [2020, 2021, 2022, 2023]  # Use multiple years for training
    all_matchup_data = []
    
    for year in train_years:
        matchup_data = prepare_matchup_data(year)
        all_matchup_data.append(matchup_data)
    
    combined_data = pd.concat(all_matchup_data, ignore_index=True)
    
    # Prepare features and labels
    X = np.stack(combined_data['features'].values)
    y = combined_data['label'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create dataset and dataloader
    dataset = NFLMatchupDataset(X_scaled, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize and train model
    input_features = X.shape[1]
    model = NFLMatchupPredictor(input_features)
    
    # Training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print("Training model...")
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss/len(dataloader):.4f} Accuracy: {accuracy:.2f}%')
    
    # Save the model and scaler
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler
    }, 'nfl_matchup_predictor.pth')
    
    # Example prediction
    current_year_pbp = nfl.import_pbp_data([2024])
    result = predict_matchup(model, scaler, 'BAL', 'CIN', current_year_pbp)
    
    if result:
        print("\nExample Prediction:")
        print(f"Predicted Winner: {result['prediction']}")
        print(f"Home Win Probability: {result['home_win_probability']:.2f}")
        print(f"Away Win Probability: {result['away_win_probability']:.2f}")

if __name__ == "__main__":
    main()