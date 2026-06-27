import os
import sys
from src.exception import CustomException
from src.logger import logging
#from configs.data_constants_config import RUL_CAP
import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_raw_path: str = os.path.join('data/raw', 'train_FD001.txt')
    test_data_raw_path: str = os.path.join('data/raw', 'test_FD001.txt')

    train_data_csv_path: str = os.path.join('data/processed', 'train_set_001.csv')
    test_data_csv_path: str = os.path.join('data/processed', 'test_set_001.csv')
    

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def inititiate_data_ingestion(self):
        logging.info("Entered data ingestion method") 
        try: 
            df_train = pd.read_csv(self.ingestion_config.train_data_raw_path, sep=r"\s+",)     #This line of code should only be changed depending on whether the source is an API or a database
            df_test = pd.read_csv(self.ingestion_config.test_data_raw_path, sep=r"\s+",)

            columns = ( ["engine_id", "cycle"] + [f"operational_setting_{i}" for i in range(1, 4)] + 
                [f"s{i}" for i in range(1, 22)])
            
            df_train.columns = columns
            df_test.columns = columns
            logging.info("Read the unprocessed train and test dataset as dataframe")

            '''max_cycles = (df_train.groupby('engine_id')['cycle'].max().rename('max_cycle'))
            df_train = df_train.join(max_cycles, on='engine_id') 
            df_train['RUL'] = df_train['max_cycle'] - df_train['cycle']
            df_train['RUL'] = df_train['RUL'].clip(upper = RUL_CAP)       #Cap RUL at RUL_CAP
            df_train.drop(columns=['max_cycle'], inplace=True)
            logging.info("Added RUL labels to training data")'''

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_csv_path), exist_ok = True)

            df_train.to_csv(self.ingestion_config.train_data_csv_path, index = False, header = True)
            df_test.to_csv(self.ingestion_config.test_data_csv_path, index = False, header = True)
            logging.info("Ingestion of data  is complete")
        
            return(
                self.ingestion_config.train_data_csv_path, 
                self.ingestion_config.test_data_csv_path, 
            )
        
        except Exception as e: 
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    obj.inititiate_data_ingestion()