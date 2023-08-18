import os
import sys
from dataclasses import dataclass
from sklearn.neural_network import BernoulliRBM
import numpy as np
import pickle

from src.exception import CustomException
from src.logger import logging

from src.utils import  load_object, save_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.model = None  # Initialize model attribute

    def initiate_model_trainer(self, train_data_path):
        try:
            if os.path.exists(self.model_trainer_config.trained_model_file_path):
                # print('exist')
    #             load_object(self.model_trainer_config.trained_model_file_path)
    #             Load the model if it's already trained
                loaded_preprocessor = load_object('artifacts/preprocessor.pkl')
                train_data = loaded_preprocessor.get_data(train_data_path)
                
                with open(self.model_trainer_config.trained_model_file_path, 'rb') as model_file:
                    self.model = pickle.load(model_file)
                # self.model = rbm.fit(train_data)

                res = self.model.gibbs(np.array(train_data))
                return res
            else:
                # print('not exist')
                loaded_preprocessor = load_object('artifacts/preprocessor.pkl')
                train_data = loaded_preprocessor.get_data(train_data_path)

                rbm = BernoulliRBM(n_components=100, learning_rate=0.01, n_iter=20, random_state=123, batch_size=200, verbose=True)
                self.model = rbm.fit(train_data)

                res = self.model.gibbs(np.array(train_data))
                
                # Save the trained model
    #             with open(self.model_trainer_config.trained_model_file_path, 'wb') as model_file:
    #                 pickle.dump(self.model, model_file)
                save_model(filepath = self.model_trainer_config.trained_model_file_path, obj = self.model)
        
                return (res,
                        self.model_trainer_config.trained_model_file_path)
        except Exception as e:
            raise CustomException(e,sys)
        
    def get_res(self, data_path):
        try:

            if self.model is None:
                # Load or train the model if it doesn't exist
                self.initiate_model_trainer(data_path)

            loaded_preprocessor = load_object('artifacts/preprocessor.pkl')
            test_data = loaded_preprocessor.get_data(data_path)

            res = self.model.gibbs(np.array(test_data))
            return res
        
        except Exception as e:
            raise CustomException(e, sys)