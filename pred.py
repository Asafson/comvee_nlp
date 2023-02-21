from datetime import datetime

import joblib
import pandas as pd

model = joblib.load('/Users/asaflev/Desktop/model.joblib')


def pred(path):
    df = pd.read_csv(path)
    df = df[['Title']].rename(columns={'Title': 'text'})
    df.dropna(subset='text', inplace=True)
    start = datetime.now()
    print('Starting predictions')
    df['pred'] = model.model.predict(df.reset_index()['text'])
    print(f'Time to predict {datetime.now() - start}')
    return df


output = pred('/Users/asaflev/Desktop/Centrical Intail List.csv')
