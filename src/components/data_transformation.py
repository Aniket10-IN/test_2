import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.pivot_column = None
        self.index_column = None
        self.value_column = None

    def get_data_transformation_object(self, input_data):
        try:
            # if not all([self.pivot_column, self.index_column, self.value_column]):
            # raise ValueError("pivot_column, index_column, and value_column must be set before calling preprocess.")
            # logging.info('data_transformation_started')

            pivoted_data = input_data.pivot(index=self.index_column, columns=self.pivot_column, values=self.value_column).fillna(0)
            pivoted_data.div(5.0)
            pivoted_data.reset_index(drop = True).T
            logging.info('data_transformation_started')

            # save_object(
            #     filepath = self.data_transformation_config.preprocessor_obj_file_path
            # )
            return pivoted_data

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, data_path):
        try:
            df = pd.read_csv(data_path)

            logging.info('read data done')

            preprocessing_obj = DataTransformation()

            logging.info('applying preprocessing obj on input data')

            preprocessing_obj.pivot_column = 'userId'
            preprocessing_obj.index_column = 'productId'
            preprocessing_obj.value_column = 'score'

            logging.info('data feeded to preprocessor object')

            preprocessed_data = preprocessing_obj.get_data_transformation_object(df)

            logging.info('preprocessing done')

            save_object(
                filepath=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                preprocessed_data,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)

        
        