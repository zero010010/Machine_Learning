def EDA(data):
  """Performs exploratory data analysis on a pandas DataFrame."""
  df = pd.read_csv('/Users/miguelopez/Desktop/Machine Learning/carbon_data.csv')
  data = df
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  
  print(data.shape)
  
  print(data.dtypes)
  
  print(data.head())
  
  print(data.tail())
  
  print(data.info())
  
  print(data.describe().T)
  
  data.hist(bins=50, figsize=(20,15))
  plt.show()
  
  for col in data.columns:
    print(col)
    print(data[col].unique())
    print("")
