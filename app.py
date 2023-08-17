import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
from src.logger import logging
from src.pipeline.predict_pipeline import Predictpipeline
from src.pipeline.training_pipeline import TrainModel
import json


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# @app.route("/")
# def index():
#     return render_template("index.html")

@app.route('/', methods=['GET', 'POST'])
def predict_data():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):

            # Save the file
            filename = os.path.join(app.config['UPLOAD_FOLDER'])
            file.save(filename)

            # Read the data
            data = pd.read_csv(filename)
            logging.info("Data loaded successfully")
            pipeline = Predictpipeline()

            # Make predictions and print
            prediction = pipeline.predict(data)
            logging.info("Prediction completed")
            return render_template('predictions.html', predictions=json.dumps(prediction.tolist()))
    return render_template('upload.html')


@app.route('/train', methods=['GET', 'POST'])
def train_model():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'])
            file.save(filename)
            
            train_model = TrainModel(data_path=filename)
            model = train_model.train_model()


            details = {
                "Name" : model, 
                "model_params" : pd.DataFrame(model.get_params(), index=['Value']).T
            }

            return render_template('model_params.html', model_params=details)
    return render_template('train.html')


if __name__ == '__main__':
    app.run(debug=True)

