from fastapi import FastAPI
from pydantic import BaseModel
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import numpy as np

app = FastAPI()

class Item(BaseModel):
    data: list

@app.post("/predict")
def predict(item: Item):
    df_f = pd.DataFrame(item.data)

    model_sarimax = SARIMAX(df_f['Power (kW)'], order=(2, 2, 1), seasonal_order=(1, 1, 1, 12))
    results = model_sarimax.fit()

    prediction = results.predict(start=len(df_f), end=len(df_f), dynamic=False)

    return {'prediction': prediction.tolist()}
