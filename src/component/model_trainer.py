import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models


@dataclass
class ModeltrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModeltrainerConfig()

    def train_model(self,train_array,test_array):

        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                            "Random Forest": RandomForestRegressor(),
                            "Decision Tree": DecisionTreeRegressor(),
                            "Gradient Boosting": GradientBoostingRegressor(),
                            "Linear Regression": LinearRegression(),
                            "XGBRegressor": XGBRegressor(random_state=42,
                                        n_jobs=-1,
                                        objective="reg:squarederror"),
                            "CatBoosting Regressor": CatBoostRegressor(verbose=True,random_state=42),
                            "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
                        }
            params={
                            "Decision Tree": {
                                'criterion':['squared_error', 'absolute_error'],
                                "max_depth": [None, 5, 10, 20],
                                "min_samples_split": [2, 5, 10]
                            },
                            "Random Forest":{
                                     "n_estimators": [100, 200],
                                     "max_depth": [None, 10, 20],
                                     "min_samples_split": [2, 5],
                                    "max_features": ["sqrt", 1.0]
                            },
                            "Gradient Boosting":{
                                        "n_estimators": [100, 200],
                                        "learning_rate": [0.05, 0.1],
                                        "max_depth": [3, 5],
                                       "subsample": [0.8, 1.0]

                            },
                            "Linear Regression":{},
                            "XGBRegressor":{
                                "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 6],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
                            },
                            "CatBoosting Regressor":{
                                "iterations": [100, 200],
                                "depth": [6, 8],
                                "learning_rate": [0.05, 0.1]
                            },
                            "AdaBoost Regressor":{
                                "n_estimators": [50, 100, 200],
                                "learning_rate": [0.05, 0.1, 0.5]
                            }
                            
                        }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                                         models=models,param=params)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square,model_report,best_model_name
            
        except Exception as e:
            raise CustomException(e,sys)

