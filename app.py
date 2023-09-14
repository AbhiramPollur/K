import os
import io
import csv
from io import StringIO
from sklearn.metrics import f1_score, precision_score, accuracy_score
import flask
import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from sklearn.metrics import classification_report
import time
from datetime import datetime

app = Flask(__name__)

def transform(text_file_contents):
    return text_file_contents.replace("=", ",")

model_path = 'models/KNC.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as model_file:
        knn_classifier = joblib.load(model_file)
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
            stream = io.StringIO(csv_file.stream.read().decode("UTF8"), newline=None)
            csv_input = csv.reader(stream)
            print(csv_input)
            for row in csv_input:
                print(row)

            stream.seek(0)
            result = transform(stream.read())
            df = pd.read_csv(StringIO(result))
            x = df.drop('label', axis=1)
            x['timestamp'] = pd.to_datetime(x['timestamp']).astype('int64') // 10**9
            y = df[['label']]
            predictions = knn_classifier.predict(x)

            knc_cr = classification_report(predictions, y)
            f1_ = f1_score(predictions, y, average='weighted')
            precision = precision_score(predictions, y, average='weighted')
            accuracy = accuracy_score(predictions, y)

            return render_template('result.html', f1_score=f1_, precision=precision, accuracy=accuracy)
            response = flask.make_response(x.to_csv())
            response.headers["Content-Disposition"] = "attachment; filename=result.csv"
            return render_template('result.html', prediction=knc_cr)

    else:
        timestamp = request.form['time']
        if timestamp:
            input_datetime = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M')
            unix_timestamp = int(input_datetime.timestamp())
        else:
            unix_timestamp = int(time.time())
        thighx = float(request.form['thigh_x'])
        thighy = float(request.form['thigh_y'])
        thighz = float(request.form['thigh_z'])
        backx = float(request.form['back_x'])
        backy = float(request.form['back_y'])
        backz = float(request.form['back_z'])

        input_data = [[unix_timestamp, thighx, thighy, thighz, backx, backy, backz]]
        prediction = knn_classifier.predict(input_data)
        if prediction == 1:
            return render_template('result.html', prediction='Walking')
        elif prediction == 2:
            return render_template('result.html', prediction='Running')
        elif prediction == 3:
            return render_template('result.html', prediction='Shuffling')
        elif prediction == 4:
            return render_template('result.html', prediction='Ascending Stairs')
        elif prediction == 5:
            return render_template('result.html', prediction='Descending Stairs')
        elif prediction == 6:
            return render_template('result.html', prediction='Standing')
        elif prediction == 7:
            return render_template('result.html', prediction='Sitting')
        elif prediction == 8:
            return render_template('result.html', prediction='Lying')
        elif prediction == 13:
            return render_template('result.html', prediction='Cycling in Sit position')
        elif prediction == 14:
            return render_template('result.html', prediction='Cycling in Standing position')
        elif prediction == 130:
            return render_template('result.html', prediction='Cycling in Sit position(Inactive)')
        elif prediction == 140:
            return render_template('result.html', prediction='Cycling in Standing position(Inactive)')
        else:
            return render_template('result.html', prediction='Unknown')



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
                joblib.dump(knn_classifier, new_model_file)

            if os.path.exists(new_model_path):
                os.remove(model_path)
                os.rename(new_model_path, model_path)

            if os.path.exists(temp_path):
                os.remove(temp_path)

            return "Model retrained successfully."

    return render_template('retrain.html')


if __name__ == '__main__':
    app.run(debug=True)
