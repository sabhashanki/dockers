from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger


app = Flask(__name__)
model = pickle.load(open('random_model.pkl','rb'))

@app.route('/')
def welcome():
    return 'Welcome to All'

@app.route('/predict')
def predict_note_authentication():
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = model.predict([[variance, skewness, curtosis, entropy]])
    return 'Predicted value is '+ str(prediction)

@app.route('/predict_file', methods = ['POST'])
def predict_file():
    df = pd.read_csv(request.files.get('file'))
    prediction = model.predict(df)
    return 'Predicted values for the file is' + str(list(prediction))

if __name__ == '__main__':
    app.run()