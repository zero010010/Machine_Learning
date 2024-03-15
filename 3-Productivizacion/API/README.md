# Machine Learning API

This project creates a machine learning API using FastAPI.
This is only a test API for demonstration purposes.

## Endpoints

The API contains the following endpoints:

- `/v1/head`: Returns the first 5 rows of the dataset
- `/v1/setup`: Splits the data into train and test sets, saves them to CSV files, and returns a confirmation message 
- `/v1/train`: Trains a model on the training set, saves the model file, and returns training metrics
- `/v1/predict`: Accepts input data, loads the saved model, makes a prediction, and returns the predicted value
- `/v1/metrics`: Loads the saved model and test set, calculates evaluation metrics, and returns them

## Usage

The API can be started with:

uvicorn app:app --reload



The endpoints accept and return JSON data.

**Example request:**
http://127.0.0.1:8000/v1/metrics


**Example response:**

{"R2_Score":0.9584512225663736,"MSE":0.0006701647046137935,"MAE":0.008708276545657291}


The API provides a simple way to access key machine learning functions for a deployed model.
