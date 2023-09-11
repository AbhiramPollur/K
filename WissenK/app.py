import os
import pickle
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from sklearn.metrics import classification_report

app = Flask(__name__)

model_path = 'models/KNC.pkl'
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

    if 'csv_file' in request.files:
        csv_file = request.files['csv_file']
        if csv_file.filename != '':
            df = pd.read_csv(csv_file)
            predictions = knn_classifier.predict(df)

            report = classification_report(df['labels'], predictions)
            return render_template('result.html', report=report)

    else:
        time = float(request.form['timestamp'])
        thighx = float(request.form['thigh_x'])
        thighy = float(request.form['thigh_y'])
        thighz = float(request.form['thigh_z'])
        backx = float(request.form['back_x'])
        backy = float(request.form['back_y'])
        backz = float(request.form['back_z'])

        input_data = [[time, thighx, thighy, thighz, backx, backy, backz]]
        prediction = knn_classifier.predict(input_data)  # Replace 'input_data' with your preprocessed data
        return render_template('result.html', prediction=prediction)


@app.route('/retrain', methods=['GET', 'POST'])
def retrain():
    global knn_classifier

    if request.method == 'POST':
        new_training_data = request.files['new_training_data']

        if new_training_data:
            temp_path = 'temp_new_training_data.csv'
            new_training_data.save(temp_path)

            df_new_data = pd.read_csv(temp_path)

            X_new = df_new_data.drop('label', axis=1)
            X_new['timestamp'] = pd.to_datetime(X_new['timestamp']).astype('int64') // 10**9
            y_new = df_new_data[['labels']]

            knn_classifier.fit(X_new, y_new)

            new_model_path = 'models/new_KNC_model.pkl'
            with open(new_model_path, 'wb') as new_model_file:
                pickle.dump(knn_classifier, new_model_file)

            if os.path.exists(new_model_path):
                os.remove(model_path)
                os.rename(new_model_path, model_path)

            if os.path.exists(temp_path):
                os.remove(temp_path)

            return "Model retrained successfully."

    return render_template('retrain.html')


if __name__ == '__main__':
    app.run(debug=True)
