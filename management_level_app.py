from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, request, Response

from constants import classes_num_to_str

app = Flask(__name__)
model = joblib.load('/Users/asaflev/Desktop/model.joblib')


@app.route('/predict', methods=['POST'])
def process():
    json_data = request.get_json()
    df = pd.DataFrame(json_data)
    df.dropna(subset='text', inplace=True)
    df['pred'] = model.model.predict(df.reset_index()['text'])
    df['pred_str'] = df['pred'].map(classes_num_to_str)
    return Response(df.to_json(orient="records"), mimetype='application/json')


if __name__ == '__main__':
    app.run()
