
# para lanzarlo en una consola

# uvicorn my_app: app --reload


from fastapi import FastAPI # importamos la clase FastAPI
import pandas as pd
import json 
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor   
import pickle 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

app = FastAPI()# FastAPI instance


@app.get("/")# root  
def welcome(): # welcome message
    return {"Hello, Welcome to my API"}  
    


# "v1/head": Endpoint que devuelve las 5 primeras filas de nuestros datos

@app.get("/v1/head")
def head_api():# definimos la función para que devuelva las cinco primeras filas de df
    df = pd.read_csv("/Users/miguelopez/Desktop/API/data/processed/df_numerical_scaled_dummies.csv")   
    df_=df.head(5)
    return json.loads(df_.to_json(orient="records"))    


# v1/setup": Un endpoint que hace todas las transformaciones que vuestros datos necesiten. 
# Cuando las tengas todas, guarda X_train, X_test, y_train e y_test en un .csv, para poder utilizar luego estos datos en otro endpoint. 
# La API tiene que devolver un mensajito que diga que todo está bien
    
@app.get("/v1/setup")
def setup_api():    
    df = pd.read_csv("/Users/miguelopez/Desktop/API/data/processed/df_final.csv")
    X = df.drop('CarbonEmission', axis=1)
    y = df['CarbonEmission']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train.to_csv('/Users/miguelopez/Desktop/API/data/processed/X_train.csv', index=False)
    X_test.to_csv('/Users/miguelopez/Desktop/API/data/processed/X_test.csv', index=False)
    y_train.to_csv('/Users/miguelopez/Desktop/API/data/processed/y_train.csv', index=False)
    y_test.to_csv('/Users/miguelopez/Desktop/API/data/processed/y_test.csv', index=False)
    return {"message": "Data Splitted and Saved!"}


@app.get("/v1/train")
def train_api():
    # cargar los datos
  X_train = pd.read_csv('/Users/miguelopez/Desktop/API/data/processed/X_train.csv')
  y_train = pd.read_csv('/Users/miguelopez/Desktop/API/data/processed/y_train.csv') 
    #  modelo
  model = DecisionTreeRegressor()
    # train fit 
  model.fit(X_train, y_train)   
    # save model.plk
  pickle.dump(model, open('model.plk','wb'))

  # return metrics message
  return {"R2_Score":r2_score(y_train, model.predict(X_train))}


# "v1/predict": Un endpoint para obtener predicciones.
#  Esta llamada no puede ser de tipo GET, 
# debe de ser de tipo POST. 
# Lo que tienes que hacer es cargar el modelo, y llamar a la función predict del modelo pasandole los datos que le has pasado a la llamada.
#  Este endpoint debe devolver la predicción de la fila que le pasas.

@app.post("/v1/predict")
def predict(data):
    model = pickle.load(open("model.pkl", "rb"))
    return model.predict([data])[0]




# v1/metrics": Un endpoint para obtener las métricas del modelo. 
# Carga el modelo ya entrenado, carga los datos de test y obtén las métricas del rendimiento del modelo. 
# Este endpoint devuelven las métricas que has calculado en un JSON.

@app.get("/v1/metrics")
def metrics():
    model = pickle.load(open("model.pkl", "rb"))
    
    X_test = pd.read_csv("/Users/miguelopez/Desktop/API/data/processed/X_test.csv")
    y_test = pd.read_csv("/Users/miguelopez/Desktop/API/data/processed/y_test.csv")  
    
    y_pred = model.predict(X_test)
    R2 = r2_score(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)    
    MAE = mean_absolute_error(y_test, y_pred)   
    
    metrics = {"R2_Score": R2, 
               "MSE": MSE,
               "MAE": MAE,}
    
    return metrics









