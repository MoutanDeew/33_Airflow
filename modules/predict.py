import glob
import json
from datetime import datetime

import dill
import pandas as pd


def predict():
    with open('../data/models/cars_pipe_202305221126.pkl', 'rb') as file:
        best_model = dill.load(file)

    df_pred = pd.DataFrame(columns=['car_id', 'pred'])
    for filename in glob.glob('../data/test/*.json'):
        with open(filename) as fin:
            form = json.load(fin)
            df = pd.DataFrame.from_dict([form])
            y = best_model.predict(df)
            x = {'car_id': df.id, 'pred': y}
            df1 = pd.DataFrame(x)
            df_pred = pd.concat([df_pred, df1], axis=0)

    df_pred.to_csv(f'../data/predictions/preds_{datetime.now().strftime("%m%d%y%H%M")}')


if __name__ == '__main__':
    predict()
