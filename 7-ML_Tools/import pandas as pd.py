import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI
import requests
import uvicorn

app =FastAPI( )

@app.get('/')
async def root():
    return{'example':'this is an example', 'data':0}
