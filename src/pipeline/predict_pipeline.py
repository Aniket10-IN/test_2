import sys, os
import pandas as pd
from src.exception import CustomException
from src.logger import logging

from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,uid, train_data_path):
        try:

            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')

            # print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)

            # print("After Loading")
            data_scaled=preprocessor.get_data(train_data_path)
            model_res = model.gibbs(data_scaled)
            # print(model_res)

            logging.info('output probabilities obtained')

            ratings = pd.read_csv('dataset/clean_data.csv', index_col = 'Unnamed: 0')
            unique_products = pd.read_csv('dataset/unique_products.csv', index_col = 'Unnamed: 0')
            logging.info('unique products done')

            rated = ratings[ratings['userId'] == uid]['productId'].tolist()
            not_rated = unique_products[~unique_products['productId'].isin(rated)]

        #     unique_products = pd.read_csv('unique_products.csv', index_col = 'Unnamed: 0')
            unique_products['recommend'] = model_res[data_scaled.index.get_loc(uid)]
            unique_products[['title', 'category', 'recommend']].head()

            recommend_products = unique_products[unique_products.recommend==True]
            final_res = recommend_products[~recommend_products.productId.isin(rated)][['title','category','score','recommend',]].head(10)
            logging.info('inference done')
            # return final_res.to_html(classes='table table-striped', index=False)
            return final_res
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(
            self,
            user_id : str):
        self.user_id = user_id

    def get_data_input(self):
        
        return self.user_id
    

if __name__ == "__main__":
    predict_pipe = PredictPipeline()
    output = predict_pipe.predict('A2S9PZCNEJEF3Y', 'dataset/clean_data.csv')
    # print(output.shape)
    # print(output)
    # cust_input = CustomData('A2S9PZCNEJEF3Y')
    # output = cust_input.get_data_input()
    # print(output)