import os
import sys
import pandas as pd
import logging



from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion has started')
        try:
            df = pd.read_csv('dataset/clean_data.csv')
            logging.info('Reading data')
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok= True)
            df.to_csv(self.ingestion_config.raw_data_path, header= True, index = False )


            logging.info('data ingestion completed')

            return (self.ingestion_config.raw_data_path)


        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    preprocessed_data = data_transformation.initiate_data_transformation(train_data)

