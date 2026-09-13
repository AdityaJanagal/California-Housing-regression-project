# California Housing Price Prediction

An end-to-end Machine Learning project that predicts California housing prices using multiple regression algorithms. The project includes data preprocessing, feature scaling, model training, model comparison, evaluation, and deployment using Flask.

## Project Overview

The goal of this project is to build a machine learning model capable of predicting median house values based on demographic and housing-related features.

Multiple regression algorithms were trained and evaluated to identify the model with the best predictive performance.

## Dataset

The project uses the California Housing dataset.

### Features

| Feature | Description |
|---|---|
| MedInc | Median income in the block |
| HouseAge | Median house age |
| AveRooms | Average number of rooms per household |
| AveBedrms | Average number of bedrooms per household |
| Population | Block population |
| AveOccup | Average number of household members |
| Latitude | Latitude of the block |
| Longitude | Longitude of the block |

### Target

**MedHouseVal** – Median house value for California districts.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- CatBoost
- Flask
- Matplotlib
- Seaborn
- Git
- GitHub

## Machine Learning Workflow

```text
Dataset
   ↓
Data Ingestion
   ↓
Data Preprocessing
   ↓
Missing Value Imputation
   ↓
Feature Scaling
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Best Model Selection
   ↓
Flask Web Application
   ↓
Prediction
