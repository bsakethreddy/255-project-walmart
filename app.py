import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
fet = pd.read_csv('joined_data_refined.csv')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    store = request.form.get('store')
    dept = request.form.get('dept')
    date = request.form.get('date')
    isHoliday = request.form.get('isholiday')

    X_test = pd.DataFrame({'Store': [21], 'Dept': [22]})

    print("X_test = ", X_test.head())
    print("type of X_test = ", type(X_test))
    y_pred = loaded_model.predict(X_test)
    print("predict = ", store, dept, date, isHoliday, y_pred)
    return y_pred
    pass

# result = loaded_model.score(X_test, Y_test)
# print(result)
# return render_template('index.html', output=result)

# @app.route('/submit', methods=['POST'])
# def getValues():


if __name__ == "__main__":
    app.run(debug=True)
