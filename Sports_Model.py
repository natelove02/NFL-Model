import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from datetime import datetime
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import csv
import time


df_SeasonOffense = pd.read_csv("TeamDefense.csv",skiprows = [0,34,35,36], header = 0)
df_SeasonDefense = pd.read_csv("TeamOffense.csv",skiprows = [0,34,35,36], header = 0)
df_WeeklyGames = pd.read_csv("FullWeeklyGames.csv", skiprows = [273], usecols=[0, 4, 6, 8, 9, 10, 11, 12, 13], header = 0)
df_WeeklyGames = df_WeeklyGames.iloc[:-13]

#ColumnNamesOff = df_SeasonOffense.iloc[1]
#ColumnNamesDef = df_SeasonDefense.iloc[1]



df_SeasonOffense = df_SeasonOffense.sort_values(by="Tm").reset_index(drop=True)
df_SeasonDefense = df_SeasonDefense.sort_values(by="Tm").reset_index(drop=True)
#df_WeeklyGames = df_WeeklyGames.sort_values(by="Winner/tie").reset_index(drop=True)



team_map = {team: idx for idx, team in enumerate(df_SeasonOffense["Tm"].unique())}
df_WeeklyGames['Loser/tie'] = df_WeeklyGames['Loser/tie'].map(team_map)
df_WeeklyGames['Winner/tie'] = df_WeeklyGames['Winner/tie'].map(team_map)
df_SeasonOffense['Tm'] = df_SeasonOffense['Tm'].map(team_map)
df_SeasonDefense['Tm'] = df_SeasonDefense['Tm'].map(team_map)

team_column = df_SeasonOffense['Tm']
teams1 = df_WeeklyGames['Winner/tie']
teams2 = df_WeeklyGames['Loser/tie']
df_WeeklyGames = df_WeeklyGames.drop(columns=['Winner/tie','Loser/tie','Week'])
df_SeasonOffense = df_SeasonOffense.drop(columns=['Rk','Tm'])
df_SeasonDefense = df_SeasonDefense.drop(columns=['Rk','Tm'])

#df_SeasonOffense = df_SeasonOffense.drop(index=[0,1])
#df_SeasonDefense = df_SeasonDefense.drop(index=[0,1])


df_WeeklyGames_NormalizedZ = (df_WeeklyGames - df_WeeklyGames.mean()) / df_WeeklyGames.std()
df_SeasonOffense_NormalizedZ = (df_SeasonOffense - df_SeasonOffense.mean()) / df_SeasonOffense.std() #Z Score normalization
df_SeasonDefense_NormalizedZ = (df_SeasonDefense - df_SeasonDefense.mean()) / df_SeasonDefense.std() #z Score normalizaiton

df_SeasonDefense_NormalizedZ['Tm'] = team_column
df_SeasonOffense_NormalizedZ['Tm'] = team_column
df_WeeklyGames_NormalizedZ['Winner/tie'] = teams1
df_WeeklyGames_NormalizedZ['Loser/tie'] = teams2


#df_WeeklyGames.index=df_WeeklyGames.set_index(['Winner/tie','Loser/tie'])
#df_WeeklyGames.index=df_WeeklyGames.index.astype(int)


full_team_data = np.stack([df_SeasonOffense_NormalizedZ.values, df_SeasonDefense_NormalizedZ.values], axis=1)

num_stats = df_SeasonDefense_NormalizedZ.shape[1]
full_team_data = full_team_data.reshape(-1,2, num_stats,1)

print(full_team_data.shape)

def GameDataPrep(df_WeeklyGames, df_SeasonOffense_NormalizedZ, df_SeasonDefense_NormalizedZ):
  X_games = []
  Y_labels = []
  reversed_team_map = {v:k for k,v in team_map.items()}
  for index, row in df_WeeklyGames_NormalizedZ.iterrows():
    winner_team = (row['Winner/tie'])
    loser_team = (row['Loser/tie'])
    #print("Team Map Keys:", team_map.keys())
    #print("Trying to access key:", winner_team)
    #print(df_WeeklyGames_NormalizedZ['Winner/tie'])
    winner_id = reversed_team_map[winner_team]
    loser_id = reversed_team_map[loser_team]

    print(df_SeasonDefense_NormalizedZ.head())
    winner_offense = df_SeasonOffense_NormalizedZ.loc[winner_id]
    winner_defense = df_SeasonDefense_NormalizedZ.loc[winner_id].values
    loser_offense = df_SeasonOffense_NormalizedZ.loc[loser_id].values
    loser_defense = df_SeasonDefense_NormalizedZ.loc[loser_id].values

    winner_data = [row['PtsW'], row['YdsW'], row['TOW']]
    loser_data = [row['PtsL'], row['YdsL'], row['TOL']]

    X_Game = [
        np.concatenate((winner_data, winner_offense, winner_defense), axis=None),
        np.concatenate((loser_data, loser_offense, loser_defense), axis=None)
    ]
    
    X_games.append(X_Game)

    Y_labels.append(1)

  return np.array(X_games), np.array(Y_labels)

x_games, y_labels = GameDataPrep(df_WeeklyGames, df_SeasonOffense_NormalizedZ, df_SeasonDefense_NormalizedZ)
print(x_games.shape)
print(y_labels.shape)

class SportsCNN(nn.Module):
  def __init__(self):
    super(SportsCNN, self).__init__()

    #2 channels for Off and Def, 32 filters for each team.
    self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(1,num_stats),stride =1 )
    self.pool = nn.MaxPool2d(kernel_size=(1,2), stride = 2)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,2), stride = 1)
    self.fc1 = nn.Linear(in_features=64*num_stats, out_features=128)
    self.fc2 = nn.Linear(in_features=128, out_features=2)

  def forward(self, x):
    x = self.pool(torch.relu(self.conv1(x)))
    x = self.pool(torch.relu(self.conv2(x)))
    x = x.view(-1 , 64*num_stats)
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    return x

model = SportsCNN()

print(model)
