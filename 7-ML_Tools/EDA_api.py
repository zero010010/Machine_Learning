import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI
import requests
import uvicorn

# Entry point for the API
app = FastAPI()

def EDA(df):
  """Performs exploratory data analysis on a pandas DataFrame."""
  
  
  print(df.shape)
  
  print(df.dtypes)
  
  print(df.head())
  
  print(df.tail())
  
  print(df.info())
  
  print(df.describe().T)
  
  df.hist(bins=50, figsize=(20,15))
  plt.show()
  
  for col in df.columns:
    print(col)
    print(df[col].unique())
    print("")


@app.post("/eda/")
async def run_eda(df: pd.DataFrame):
    EDA(df)
    return {"message": "EDA complete"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


# Testing 
# df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

# response = requests.post("http://127.0.0.1:8000/eda/", json=df.to_dict())
# print(response.text)  