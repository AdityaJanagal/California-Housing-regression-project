from src.component.data_injestion import DataIngestion
import pandas as pd

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

df = pd.read_csv(train_data)
print(df.info())