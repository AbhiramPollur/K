import os
import pickle
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from sklearn.metrics import classification_report

app = Flask(__name__)

# Load the trained model
model_path = 'models/knn_model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as model_file:
        knn_classifier = pickle.load(model_file)
else:
    knn_classifier = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if knn_classifier is None:
        return "Model not found. Please train the model first."

    # Get user input
    if 'csv_file' in request.files:
        csv_file = request.files['csv_file']
        if csv_file.filename != '':
            df = pd.read_csv(csv_file)
            predictions = knn_classifier.predict(df)

            report = classification_report(df['labels'], predictions)
            return render_template('result.html', report=report)

    else:
        time = float(request.form['time'])
        thighx = float(request.form['thighx'])
        thighy = float(request.form['thighy'])
        thighz = float(request.form['thighz'])
        backx = float(request.form['backx'])
        backy = float(request.form['backy'])
        backz = float(request.form['backz'])

        input_data = [[time, thighx, thighy, thighz, backx, backy, backz]]
        prediction = knn_classifier.predict(input_data)  # Replace 'input_data' with your preprocessed data
        return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)