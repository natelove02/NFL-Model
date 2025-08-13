# NFL Model: Data Pipeline & Betting Prediction

[![GitHub Project Board](https://img.shields.io/badge/Project-Board-blue?logo=github)](https://github.com/natelove02/NFL-Model/projects)

## Overview
This project provides a complete pipeline for scraping, cleaning, merging, and engineering NFL game and betting data, culminating in advanced predictive modeling for NFL outcomes. The workflow is modular, reproducible, and designed for both exploratory analysis and production-ready predictions.

Project management and agile workflow are tracked on the [GitHub Project Board](https://github.com/natelove02/NFL-Model/projects).

## Features
- **Data Scraping & Cleaning:** Automated collection and cleaning of NFL game logs and betting odds from public sources.
- **Feature Engineering:** Creation of advanced features such as rolling averages, ELO ratings, and matchup statistics.
- **Exploratory Analysis:** Visualizations and summary statistics for team and league trends.
- **Predictive Modeling:** Machine learning models (PyTorch neural networks, XGBoost) for game outcome prediction and matchup analysis.
- **Reproducible Notebooks:** All steps are documented and executable in Jupyter Notebooks for transparency and ease of use.

## Main Notebooks
- `data_cleaning_and_scraping_.ipynb`: Scrapes, cleans, and engineers features from NFL data.
- `NFL_Betting.ipynb`: Builds and evaluates predictive models for NFL games, including interactive prediction and matchup insights.

## Requirements
- Python 3.8+
- Conda (recommended) or pip
- Jupyter Notebook
- pandas, numpy, matplotlib, seaborn
- scikit-learn, xgboost, torch
- lxml (for HTML parsing)

## Setup
1. Clone the repository:
   ```bash
   git clone git@github.com:natelove02/NFL-Model.git
   cd NFL-Model
   ```
2. Create and activate the conda environment:
   ```bash
   conda create -n nflmodel python=3.13
   conda activate nflmodel
   ```
3. Install dependencies:
   ```bash
   conda install pandas numpy matplotlib seaborn scikit-learn lxml
   pip install torch xgboost
   ```
4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

## Usage
- Run `data_cleaning_and_scraping_.ipynb` to generate cleaned and feature-rich NFL datasets.
- Run `NFL_Betting.ipynb` to train models and make predictions on NFL games.

## Project Structure
- `data_cleaning_and_scraping_.ipynb` — Data pipeline and feature engineering
- `NFL_Betting.ipynb` — Modeling and prediction
- `offense.csv`, `defense.csv`, `NFLstats.csv` — Example data files

## License
MIT License

## Author
Nate Love
