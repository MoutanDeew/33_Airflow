import glob
import json
import os
from datetime import datetime

import dill
import pandas as pd

path = os.environ.get('PROJECT_PATH', '.')


def predict(): #функция дисириализации лучшей последней ML-модели и предсказания целевой переменной на новых JSON данных
    latest_model = sorted(os.listdir(f'{path}/data/models'))[-1]
    with open(f'{path}/data/models/{latest_model}', 'rb') as file:
        best_model = dill.load(file)

    df_pred = pd.DataFrame(columns=['car_id', 'pred'])
    for filename in glob.glob(f'{path}/data/test/*.json'):
        with open(filename) as fin:
            form = json.load(fin)
            df = pd.DataFrame.from_dict([form])
            y = best_model.predict(df)
            x = {'car_id': df.id, 'pred': y}
            df1 = pd.DataFrame(x)
            df_pred = pd.concat([df_pred, df1], axis=0)

    df_pred.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%m%d%y%H%M")}')


if __name__ == '__main__':
    predict()
