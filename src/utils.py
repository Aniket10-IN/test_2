import os
import sys
import pickle
from src.exception import CustomException

def save_preprocessor(filepath, obj):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok = True)


        with open(f'{dir_path}/preprocessor.pkl', "wb") as file_obj:
            pickle.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)

def save_model(filepath, obj):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok = True)


        with open(f'{dir_path}/model.pkl', "wb") as file_obj:
             pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def save_object(filepath, obj):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok = True)


        with open(dir_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)