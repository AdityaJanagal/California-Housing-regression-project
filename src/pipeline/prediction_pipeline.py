import pandas as pd
import os
from src.exception import CustomException
from src.utils import load_object 
from src.logger import logging
import sys


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,
                 MedInc: float,
                 HouseAge: float,
                 AveRooms: float,
                 AveBedrms: float,
                 Population: float,
                 AveOccup: float,
                 Latitude: float,
                 Longitude: float):
        
        self.MedInc = MedInc
        self.HouseAge = HouseAge
        self.AveRooms = AveRooms
        self.AveBedrms = AveBedrms
        self.Population = Population
        self.AveOccup = AveOccup
        self.Latitude = Latitude
        self.Longitude = Longitude

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "MedInc": [self.MedInc],
                "HouseAge": [self.HouseAge],
                "AveRooms": [self.AveRooms],
                "AveBedrms": [self.AveBedrms],
                "Population": [self.Population],
                "AveOccup": [self.AveOccup],
                "Latitude": [self.Latitude],
                "Longitude": [self.Longitude]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Dataframe Gathered")
            return df
        except Exception as e:
            raise CustomException(e, sys)