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


Data Preprocessing

The following preprocessing steps were performed:

Loaded and explored the dataset using Pandas.
Separated input features and target variable.
Handled missing values using median imputation.
Applied StandardScaler for feature scaling.
Split the dataset into training and testing sets.
Prepared the processed data for machine learning models.
Models Implemented

The following regression algorithms were trained and evaluated:

Linear Regression
Decision Tree Regressor
Random Forest Regressor
Gradient Boosting Regressor
XGBoost Regressor
CatBoost Regressor
AdaBoost Regressor
Model Performance

The models were evaluated using the R² (R-squared) score.

Model	R² Score
XGBoost	0.8442
CatBoost	0.8372
Gradient Boosting	0.8330
Random Forest	0.8146
Decision Tree	0.6696
Linear Regression	0.5911
AdaBoost	0.5748
Best Model

XGBoost Regressor achieved the highest R² score of 0.8442 among the evaluated models.

This indicates that XGBoost provided the strongest predictive performance on the test dataset among the models evaluated.

Model Evaluation

Model performance was compared using:

R² Score
Predicted vs. actual performance
Comparison of different regression algorithms

The model with the best test performance was selected for the final prediction pipeline.

Project Structure
California-Housing-regression-project/
│
├── artifacts/
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── notebook/
│   └── data_analysis.ipynb
│
├── src/
│   ├── components/
│   ├── pipelines/
│   └── ...
│
├── templates/
│   └── index.html
│
├── app.py
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore

The exact project structure may vary depending on the latest version of the repository.

Deployment

The trained machine learning model was integrated into a Flask web application.

The application accepts housing-related input features and uses the trained model and preprocessing pipeline to generate a predicted house value.

Application Flow
User Input
    ↓
Flask Application
    ↓
Data Preprocessing
    ↓
Trained ML Model
    ↓
Predicted House Value
    ↓
Result Display
Installation

Clone the repository:

git clone https://github.com/AdityaJanagal/California-Housing-regression-project.git

Navigate to the project directory:

cd California-Housing-regression-project

Create a virtual environment:

python -m venv venv

Activate the virtual environment on Windows:

venv\Scripts\activate

Install the required dependencies:

pip install -r requirements.txt
Run the Application

Start the Flask application:

python app.py

Then open the local application in your browser:

http://127.0.0.1:5000/
Example Prediction Inputs

The application uses the following features for prediction:

MedInc
HouseAge
AveRooms
AveBedrms
Population
AveOccup
Latitude
Longitude

The trained preprocessing pipeline transforms the input before passing it to the selected machine learning model.

Key Learnings

Through this project, I gained practical experience in:

End-to-end machine learning workflow
Regression problems
Data preprocessing
Missing value handling
Feature scaling
Model training and comparison
Model evaluation using R²
XGBoost and CatBoost
Model serialization
Flask-based ML deployment
Building a prediction pipeline
Future Improvements
Improve model performance through hyperparameter tuning.
Add additional feature engineering techniques.
Experiment with advanced ensemble methods.
Improve the Flask application's UI/UX.
Containerize the application using Docker.
Deploy the application using a cloud platform.
Author

Aditya Janagal

B.Tech – Artificial Intelligence & Data Science

GitHub: AdityaJanagal
   ↓
Prediction
