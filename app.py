import numpy as np
import pandas as pd
from flask import Flask, render_template
import pickle

app = Flask(__name__)
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
fet = pd.read_csv('joined_data_refined.csv')

@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/predict',methods=['POST'])
# def predict():
#     result = loaded_model.score(X_test, Y_test)
#     print(result)
#     return render_template('index.html', output=result)


if __name__ == "__main__":
    app.run(debug=True)