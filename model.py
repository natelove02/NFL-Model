import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import fire
import os


class SimpleLinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=3, dropout=0.2):
        super(MLPModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(CNNBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection if dimensions don't match
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        out += identity  # Residual connection
        out = self.relu(out)
        return out

class CNNModel(nn.Module):
    def __init__(self, input_size, output_size=3, base_channels=32):
        super(CNNModel, self).__init__()
        
        # Initial projection to get started
        self.input_proj = nn.Conv1d(1, base_channels, 1)
        
        # Stack of CNN blocks with increasing channels
        self.blocks = nn.ModuleList([
            CNNBlock(base_channels, base_channels),
            CNNBlock(base_channels, base_channels * 2),
            CNNBlock(base_channels * 2, base_channels * 4),
            CNNBlock(base_channels * 4, base_channels * 4)
        ])
        
        # Global average pooling and final classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        # Reshape for conv1d: (batch_size, 1, features)
        x = x.unsqueeze(1)
        
        # Initial projection
        x = self.input_proj(x)
        
        # Pass through CNN blocks
        for block in self.blocks:
            x = block(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.squeeze(-1)  # Remove last dimension
        return self.classifier(x)    


class NFLModel:
    def __init__(self):
        self._model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def weighted_mixed_loss(self, predictions, targets):
        spread_pred, total_pred, win_pred = predictions[:, 0], predictions[:, 1], predictions[:, 2]
        spread_true, total_true, win_true = targets[:, 0], targets[:, 1], targets[:, 2]
        
        # Weight spread prediction higher since it's harder
        spread_loss = nn.HuberLoss()(spread_pred, spread_true) * 2.0  # Huber is better for outliers
        total_loss = nn.MSELoss()(total_pred, total_true)
        win_loss = nn.BCEWithLogitsLoss()(win_pred, win_true)
        
        return spread_loss + total_loss + win_loss  
    
    
    
    
    def load_data(self, data_path='final_modeling_dataset.csv'):
        """Load and prepare the NFL modeling dataset"""
        df = pd.read_csv(data_path)
        
        # Define feature columns (exclude context and target columns)
        context_cols = ['Game_ID', 'Season', 'Week', 'Date']
        target_cols = ['spread_target', 'total_target', 'win_target']
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in context_cols + target_cols]
        
        X = df[feature_cols].values
        y = df[target_cols].values
        
        print(f"Dataset shape: {df.shape}")
        print(f"Features: {len(feature_cols)}")
        print(f"Samples: {len(X)}")
        
        return X, y, feature_cols, target_cols
    
    def train_model(self, data_path='final_modeling_dataset.csv', epochs=1000, lr=0.001):
        """Train a simple linear model on NFL data"""
        # Load data
        X, y, feature_cols, target_cols = self.load_data(data_path)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)
        
        # Initialize model
        input_size = X_train_scaled.shape[1]
        output_size = y_train.shape[1]  # 3 targets: spread, total, win
        
        self._model = SimpleLinearModel(input_size, output_size).to(self.device)
        
        # Loss function and optimizer
        criterion = self.weighted_mixed_loss  # Use custom loss function
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr)

        print(f"Training model with {input_size} features -> {output_size} targets")
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Training loop
        for epoch in range(epochs):
            self._model.train()
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self._model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
        # Evaluate model
        self._model.eval()
        with torch.no_grad():
            train_predictions = self._model(X_train_tensor).cpu().numpy()
            test_predictions = self._model(X_test_tensor).cpu().numpy()

        # Calculate metrics for each target
        target_names = ['Spread', 'Total', 'Win']
        
        print("\n=== MODEL EVALUATION ===")
        for i, target_name in enumerate(target_names):
            print(f"\n{target_name} Target:")
            
            if target_name == 'Win':
                # Classification metrics for win prediction
                train_acc = accuracy_score(y_train[:, i] > 0.5, train_predictions[:, i] > 0.5)
                test_acc = accuracy_score(y_test[:, i] > 0.5, test_predictions[:, i] > 0.5)
                print(f"  Train Accuracy: {train_acc:.4f}")
                print(f"  Test Accuracy: {test_acc:.4f}")
            else:
                # Regression metrics for spread and total
                train_mse = mean_squared_error(y_train[:, i], train_predictions[:, i])
                test_mse = mean_squared_error(y_test[:, i], test_predictions[:, i])
                train_mae = mean_absolute_error(y_train[:, i], train_predictions[:, i])
                test_mae = mean_absolute_error(y_test[:, i], test_predictions[:, i])
                
                print(f"  Train MSE: {train_mse:.4f}, MAE: {train_mae:.4f}")
                print(f"  Test MSE: {test_mse:.4f}, MAE: {test_mae:.4f}")
        
        print(f"\nModel training completed!")
        return self._model


def check_data():
    """Check the dataset and print basic info"""
    nfl_model = NFLModel()
    X, y, feature_cols, target_cols = nfl_model.load_data()
    return "Data loaded successfully"

def train_model(epochs=100, lr=0.01, model_name="nfl_linear"):
    """Train the NFL model"""
    nfl_model = NFLModel()
    model = nfl_model.train_model(epochs=epochs, lr=lr)
    
    # Save model if needed
    torch.save(model.state_dict(), f"{model_name}.pth")
    print(f"Model saved to {model_name}.pth")
    #return model

def train_mlp(epochs=100, lr=0.001, model_name="nfl_mlp"):
    """Train MLP model"""
    nfl_model = NFLModel()
    nfl_model.model_type = "mlp"
    model = nfl_model.train_model(epochs=epochs, lr=lr)
    torch.save(model.state_dict(), f"{model_name}.pth")
    return f"MLP training completed! Model saved to {model_name}.pth"

def train_cnn(epochs=100, lr=0.001, model_name="nfl_cnn"):
    """Train CNN model"""
    nfl_model = NFLModel()
    nfl_model.model_type = "cnn"
    model = nfl_model.train_model(epochs=epochs, lr=lr)
    torch.save(model.state_dict(), f"{model_name}.pth")
    return f"CNN training completed! Model saved to {model_name}.pth"

def check_targets():
    """Debug the target variables"""
    df = pd.read_csv('final_modeling_dataset.csv')
    
    print(f"Home team win rate: {df['win_target'].mean():.3f}")
    print(f"Spread target stats: mean={df['spread_target'].mean():.2f}, std={df['spread_target'].std():.2f}")
    print(f"Total target stats: mean={df['total_target'].mean():.1f}, std={df['total_target'].std():.1f}")
    
    return "Target analysis complete"


def check_predictions():
    """See what the model is actually predicting"""
    # Load some recent predictions and compare to actuals
    # This will show if it's systematically over/under predicting totals
    """See what the model is actually predicting vs actuals"""
    # Load data and model
    nfl_model = NFLModel()
    X, y, feature_cols, target_cols = nfl_model.load_data()
    
    # Quick train to get predictions
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled = nfl_model.scaler.fit_transform(X_train)
    X_test_scaled = nfl_model.scaler.transform(X_test)
    
    # Load your saved model
    import torch
    nfl_model._model = SimpleLinearModel(X.shape[1], 3)
    nfl_model._model.load_state_dict(torch.load('nfl_linear.pth'))
    nfl_model._model.eval()
    
    # Get predictions
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test_scaled)
        predictions = nfl_model._model(X_tensor).numpy()
    
    print("=== PREDICTION ANALYSIS ===")
    print(f"Actual spreads: min={y_test[:, 0].min():.1f}, max={y_test[:, 0].max():.1f}, mean={y_test[:, 0].mean():.1f}")
    print(f"Predicted spreads: min={predictions[:, 0].min():.1f}, max={predictions[:, 0].max():.1f}, mean={predictions[:, 0].mean():.1f}")
    
    print(f"Actual totals: min={y_test[:, 1].min():.1f}, max={y_test[:, 1].max():.1f}, mean={y_test[:, 1].mean():.1f}")
    print(f"Predicted totals: min={predictions[:, 1].min():.1f}, max={predictions[:, 1].max():.1f}, mean={predictions[:, 1].mean():.1f}")
    
    print(f"Actual wins: {(y_test[:, 2] > 0.5).mean():.3f}")
    print(f"Predicted wins: {(predictions[:, 2] > 0.5).mean():.3f}")
    
    # Show some examples
    print("\nSample predictions vs actuals:")
    for i in range(5):
        print(f"Game {i+1}: Spread {predictions[i,0]:.1f} vs {y_test[i,0]:.1f}, Total {predictions[i,1]:.1f} vs {y_test[i,1]:.1f}")
    
    return "Prediction analysis complete"


def predict_manual(
    home_team: str = "DAL",
    away_team: str = "PHI",
    model_path: str = "nfl_linear.pth",
    # Quick-input knobs (others default to 0)
    elo_diff: float = 0.0,
    is_international: int = 0,
    season_weight: float = 0.3,
    off_pts_per_drive_vs_def_pts_per_poss: float = 0.0,
    off3d_vs_def3d: float = 0.0,
    off4d_vs_def4d: float = 0.0,
    to_rate_vs_def_to_rate: float = 0.0,
    total_off_eff_diff: float = 0.0,
    total_def_eff_diff: float = 0.0,
):
    """Manual W/L prediction using a few key inputs.

    Notes:
    - Predicts from the home team's perspective (win_target=1 means home win).
    - Unspecified features default to 0, which is a reasonable neutral baseline for diffs.
    - Uses a scaler fit on the full dataset to transform the manual row.
    """

    if not os.path.exists(model_path):
        return f"Model file '{model_path}' not found. Train first with: python model.py train --epochs=200 --lr=0.001 --model_name=nfl_linear"

    # Load dataset to recover feature columns and fit scaler
    nfl_model = NFLModel()
    X, y, feature_cols, target_cols = nfl_model.load_data()
    scaler = StandardScaler().fit(X)

    # Build a zero vector and set available inputs
    row = np.zeros(len(feature_cols), dtype=float)

    def set_feat(name: str, value: float):
        if name in feature_cols:
            row[feature_cols.index(name)] = value

    # Map provided inputs onto known columns
    set_feat('IsInternational', float(is_international))
    set_feat('Elo_Diff', float(elo_diff))
    set_feat('H_Season_Weight', float(season_weight))
    set_feat('OffPtsPerDrive_vs_DefPtsPerPoss', float(off_pts_per_drive_vs_def_pts_per_poss))
    set_feat('Off3DSuccess_vs_Def3DStop', float(off3d_vs_def3d))
    set_feat('Off4DSuccess_vs_Def4DStop', float(off4d_vs_def4d))
    set_feat('OffTORate_vs_DefTORate', float(to_rate_vs_def_to_rate))
    set_feat('Total_Offensive_Efficiency_Diff', float(total_off_eff_diff))
    set_feat('Total_Defensive_Efficiency_Diff', float(total_def_eff_diff))

    # Scale and predict
    row_scaled = scaler.transform(row.reshape(1, -1))
    model = SimpleLinearModel(input_size=len(feature_cols), output_size=3)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    with torch.no_grad():
        x_tensor = torch.from_numpy(row_scaled.astype(np.float32))
        preds = model(x_tensor).numpy().reshape(-1)

    spread_pred, total_pred, win_logit = float(preds[0]), float(preds[1]), float(preds[2])
    win_prob = float(torch.sigmoid(torch.tensor(win_logit)).item())
    pick = f"Home ({home_team})" if win_prob >= 0.5 else f"Away ({away_team})"

    print("=== MANUAL W/L PREDICTION ===")
    print(f"Home: {home_team} vs Away: {away_team}")
    print(f"Inputs -> Elo_Diff={elo_diff}, SeasonWeight={season_weight}, Intl={is_international}, OffEffDiff={total_off_eff_diff}, DefEffDiff={total_def_eff_diff}")
    print(f"Predicted home win probability: {win_prob:.3f} -> Pick: {pick}")

    return {
        'home': home_team,
        'away': away_team,
        'home_win_prob': win_prob,
        'pick': pick,
        'debug': {
            'spread_pred': spread_pred,
            'total_pred': total_pred,
        }
    }


def predict_prompt(model_path: str = "nfl_linear.pth"):
    """Interactive prompt to enter a few values and get a W/L pick."""
    try:
        home_team = input("Home team (e.g., DAL): ").strip() or "DAL"
        away_team = input("Away team (e.g., PHI): ").strip() or "PHI"
        elo_diff = float(input("Elo_Diff (home - away, e.g., 25): ") or 0.0)
        season_weight = float(input("Season weight (0-1, default 0.3): ") or 0.3)
        is_international = int(input("International game? 0/1 (default 0): ") or 0)
        total_off_eff_diff = float(input("Total Offensive Efficiency Diff (home-away, default 0): ") or 0.0)
        total_def_eff_diff = float(input("Total Defensive Efficiency Diff (home-away, default 0): ") or 0.0)
    except Exception as e:
        return f"Invalid input: {e}"

    return predict_manual(
        home_team=home_team,
        away_team=away_team,
        model_path=model_path,
        elo_diff=elo_diff,
        season_weight=season_weight,
        is_international=is_international,
        total_off_eff_diff=total_off_eff_diff,
        total_def_eff_diff=total_def_eff_diff,
    )


def main():
    fire.Fire({
        "check": check_data, 
        "check_targets": check_targets,
        "train": train_model,
        "train_mlp": train_mlp,
        "train_cnn": train_cnn,
        "check_predictions": check_predictions,
        "predict": predict_manual,
        "predict_prompt": predict_prompt,
    })

if __name__ == "__main__":
    main()
