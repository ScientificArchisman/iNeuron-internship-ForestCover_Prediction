import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
from src.logger import logging
from src.pipeline.predict_pipeline import Predictpipeline
from src.pipeline.training_pipeline import TrainModel
import json
import flasgger
from flasgger import Swagger

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

Swagger(app)

def allowed_file(filename):
    """ Check if the file is allowed to be uploaded
    Args:
        filename (str): name of the file
    Returns:
        bool: True if allowed, False otherwise"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route("/")
def index():
    return render_template("index.html")




@app.route('/predict_data', methods=['POST'])
def predict_data():
    """ Let's predict the forest cover type
    ---
    parameters:
      - in: formData
        name: file
        type: file
        required: true
        description: The CSV file to upload.
    responses:
      200:
        description: File successfully uploaded
      400:
        description: Bad request (e.g., no file provided)
    """
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


@app.route('/train', methods=['POST'])
def train_model():
    """ Let's predict the forest cover type
    ---
    parameters:
      - in: formData
        name: file
        type: file
        required: true
        description: The CSV file to upload.
    responses:
      200:
        description: File successfully uploaded
      400:
        description: Bad request (e.g., no file provided)
    """
     
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

