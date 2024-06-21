import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import os

app = Flask(__name__)

# Correct model file names
diabetes_model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler2 = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            print("POST request received")
            
            # Ensure all fields are present
            fields = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Age", "DiabetesPedigreeFunction"]
            for field in fields:
                if field not in request.form or request.form[field] == "":
                    raise ValueError(f"Missing value for {field}")
            
            Pregnancies = int(request.form["Pregnancies"])
            Glucose = int(request.form["Glucose"])
            BloodPressure = int(request.form["BloodPressure"])
            SkinThickness = int(request.form["SkinThickness"])
            Insulin = int(request.form["Insulin"])
            BMI = float(request.form["BMI"])
            Age = int(request.form["Age"])
            DiabetesPedigreeFunction = float(request.form["DiabetesPedigreeFunction"])

            input_data = (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, Age, DiabetesPedigreeFunction)
            input_data_as_numpy_array = np.asarray(input_data)
            input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
            std_data = scaler2.transform(input_data_reshaped)
            prediction = diabetes_model.predict(std_data)

            if prediction[0] == 0:
                result = "You are Not Diabetic"
            else:
                result = "You are Diabetic"

            print("Result prepared")

            return render_template('index.html', prediction_texts=f'Diabetes prediction is {result}')
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return render_template('index.html', prediction_texts=f'Error: {str(e)}')
    else:
        print("GET request received")
        return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True)
