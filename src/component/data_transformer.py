import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):

        try:
            numerical_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
            num_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
                ]
            )
            logging.info("Numerical columns standard scaling completed")

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_columns)
            ])
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

        # Print actual column names to confirm
            print("Actual DataFrame columns:", train_df.columns.tolist())
            
            preprocessor_obj=self.get_data_transformer_object()

            target_column_name="median_house_value"
            input_features=train_df.drop(columns=['median_house_value'])
            target_feature=train_df['median_house_value']

            input_features_test=test_df.drop(columns=['median_house_value'])
            target_feature_test=test_df['median_house_value']
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            train_arr=preprocessor_obj.fit_transform(input_features)
            test_arr=preprocessor_obj.transform(input_features_test)

            logging.info(f"Saved preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            return (
                train_arr,
                test_arr,
                preprocessor_obj
            )
        except Exception as e:
            raise CustomException(e,sys)
