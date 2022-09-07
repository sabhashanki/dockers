from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger


app = Flask(__name__)
Swagger(app)
model = pickle.load(open('random_model.pkl','rb'))

@app.route('/')
def welcome():
    return 'Welcome to All'

@app.route('/predict')
def predict_note_authentication():

    """Lets Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:
        - name: variance
          in: query
          type: number
          required: true

        - name: skewness
          in: query
          type: number
          required: true

        - name: curtosis
          in: query
          type: number
          required: true
          
        - name: entropy
          in: query
          type: number
          required: true
    responses:
        200:
            description: The output values

    """
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = model.predict([[variance, skewness, curtosis, entropy]])
    return 'Predicted value is '+ str(prediction)

@app.route('/predict_file', methods = ['POST'])
def predict_file():

    """Lets Authenticate the Banks Note
    This is using docstrings for specifications.
    ---
    parameters:
        - name: file
          in: formData
          type: file
          required: true

    responses:
        200:
            description: The output values
    
    """

    df = pd.read_csv(request.files.get('file'))
    prediction = model.predict(df)
    return 'Predicted values for the file is' + str(list(prediction))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 8000)