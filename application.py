from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

predict_pipeline = PredictPipeline()

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        # print("Received form data:", request.form)
        return render_template('home.html')
    
    else:
        print("Received form data:", request.form)
        # user_id = CustomData(
        #     request.form.get('user_id')
        # )

        user_id = request.form['user_id']
        print('Before Prediction')

        predict_pipeline = PredictPipeline()
        print('Mid Prediction')
        results = predict_pipeline.predict(user_id,'dataset/clean_data.csv')
        print(results)
        print("after Prediction")
        return render_template('home.html',results=results.to_html(classes='table table-striped', index=False))
    
        # return render_template('home.html',results=results.to_html(classes='table table-striped', index=False))
    # print('done')

if __name__=="__main__":
    app.run(host="0.0.0.0")