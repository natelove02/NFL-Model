# Helper to get feature columns without Elo_Diff
from model import SimpleLinearModel
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import xgboost as xgb
import fire

def get_feature_cols_no_elo(df, context_cols, target_cols):
    return [col for col in df.columns if col not in context_cols + target_cols + ['Elo_Diff']]

# Linear win model (no Elo_Diff)
def train_linear_win_noelo(data_path='final_modeling_dataset.csv', epochs=200, lr=0.001, model_name='linear_win_noelo'):
    df = pd.read_csv(data_path)
    context_cols = ['Game_ID', 'Season', 'Week', 'Date']
    target_cols = ['spread_target', 'total_target', 'win_target']
    feature_cols = get_feature_cols_no_elo(df, context_cols, target_cols)
    X = df[feature_cols].values
    y = df['win_target'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SimpleLinearModel(input_size=X_train_scaled.shape[1], output_size=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

    # Evaluate
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    with torch.no_grad():
        logits = model(X_test_tensor).numpy().flatten()
        preds = (logits > 0).astype(int)
    acc = accuracy_score(y_test, preds)
    print(f'Test Accuracy: {acc:.4f}')

    torch.save(model.state_dict(), f'{model_name}.pth')
    print(f'Model saved to {model_name}.pth')
    return model, scaler, feature_cols

# XGBoost models (no Elo_Diff)
def train_xgboost_noelo(data_path='final_modeling_dataset.csv', test_size=0.2, random_state=42):
    df = pd.read_csv(data_path)
    context_cols = ['Game_ID', 'Season', 'Week', 'Date']
    target_cols = ['spread_target', 'total_target', 'win_target']
    feature_cols = get_feature_cols_no_elo(df, context_cols, target_cols)
    X = df[feature_cols].values
    y = df[target_cols].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    models = {}
    metrics = {}
    for i, target in enumerate(target_cols):
        reg = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state
        )
        reg.fit(X_train, y_train[:, i])
        y_pred = reg.predict(X_test)
        if target == 'win_target':
            acc = accuracy_score(y_test[:, i] > 0.5, y_pred > 0.5)
            metrics[target] = {'accuracy': acc}
            print(f"{target} accuracy: {acc:.3f}")
        else:
            mse = mean_squared_error(y_test[:, i], y_pred)
            metrics[target] = {'mse': mse}
            print(f"{target} MSE: {mse:.3f}")
        models[target] = reg

    for target, model in models.items():
        model.save_model(f"xgboost_{target}_noelo.json")
    print("XGBoost models saved as xgboost_spread_target_noelo.json, xgboost_total_target_noelo.json, xgboost_win_target_noelo.json")
    return models, metrics


def main():
    fire.Fire({
        "train_linear_win_noelo": train_linear_win_noelo,
        "train_xgboost_noelo": train_xgboost_noelo,
    })

if __name__ == "__main__":
    main()