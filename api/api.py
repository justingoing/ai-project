# from typing import Any, List

# import numpy as np
# from fastapi import FastAPI, HTTPException, Body, Query, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import HTMLResponse
# import os
# import uvicorn
# import logging
# from pydantic import BaseModel, Field
# from sqlalchemy import create_engine
# import pandas as pd
# from typing import Dict, List, Optional, Union
# import json
# import ssl
# import plotly.express as px
# import plotly.graph_objs as go
# from plotly.subplots import make_subplots
# from statsmodels.formula.api import ols
# import statsmodels.api as sm


# server = FastAPI(
#     title='AI Project API',
#     description='Provide data to the app'
# )

# server.add_middleware(
#     CORSMiddleware,
#     # allow_origins=["https://www.csu-hci-experiment.online"],
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# DB_USER = "postgres"
# DB_PASSWORD = "123"
# DB_HOST = "ai_project_db"
# DB_PORT = "5432"
# DB_NAME = "postgres"

# DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# engine = create_engine(DATABASE_URL)

# def fetch_data_from_database():
#     sql = """SELECT * FROM users;"""

#     df = pd.read_sql(sql, engine)

#     return df

# @server.get("/", description="Root", summary="Hello World")
# def read_root():
#     return {"hello": "world"}

# @server.get("/users", description="Get all users", summary="Get all users")
# def get_users():
#     df = fetch_data_from_database()
#     return df.to_dict(orient="records")

# if __name__ == "__main__":

#     uvicorn.run("api:server", host="0.0.0.0",
#                 port=5000, log_level="info", reload=True)


from typing import Any, List
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import create_engine
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import uvicorn

# Load the data
data = pd.read_csv('./data/mahopac-city.csv')

# Ensure that the 'City' column is treated as a categorical variable
data['City'] = data['City'].astype('category')

# Create FastAPI instance
server = FastAPI(
    title='AI Project API',
    description='Provide data to the app'
)

server.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_USER = "postgres"
DB_PASSWORD = "123"
DB_HOST = "ai_project_db"
DB_PORT = "5432"
DB_NAME = "postgres"

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

def fetch_data_from_database():
    sql = """SELECT * FROM users;"""
    df = pd.read_sql(sql, engine)
    return df

@server.get("/", description="Root", summary="Hello World")
def read_root():
    return {"hello": "world"}

@server.get("/users", description="Get all users", summary="Get all users")
def get_users():
    df = fetch_data_from_database()
    return df.to_dict(orient="records")

# Endpoint to get scatter plot
@server.get("/scatter", description="Scatter Plot", summary="Scatter Plot", response_class=HTMLResponse)
def get_scatter():
    fig = px.scatter(data, x='Living area', y='Property price (USD)', title='Living Area vs. Property Price (USD)')
    return fig.to_html(full_html=False)

# Endpoint to get histogram
@server.get("/histogram", description="Histogram", summary="Histogram", response_class=HTMLResponse)
def get_histogram():
    fig = px.histogram(data, x='Property price (USD)', nbins=50, title='Distribution of Property Prices')
    return fig.to_html(full_html=False)

# Endpoint to get box plot
@server.get("/boxplot", description="Box Plot", summary="Box Plot", response_class=HTMLResponse)
def get_boxplot():
    fig = px.box(data, x='City', y='Property price (USD)', title='Property Prices by City - Box Plot')
    return fig.to_html(full_html=False)

# Endpoint to get violin plot
@server.get("/violinplot", description="Violin Plot", summary="Violin Plot", response_class=HTMLResponse)
def get_violinplot():
    fig = px.violin(data, x='City', y='Property price (USD)', title='Property Prices by City - Violin Plot')
    return fig.to_html(full_html=False)

# Endpoint to get PCA plot
@server.get("/pca", description="PCA Plot", summary="PCA Plot", response_class=HTMLResponse)
def get_pca():
    features = ['Living area', 'Property price (USD)', 'Lot/land area', 'Bedrooms', 'Bathrooms']
    x = data[features].dropna()
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
    fig = px.scatter(principalDf, x='principal component 1', y='principal component 2', title='PCA of Property Data')
    return fig.to_html(full_html=False)

if __name__ == "__main__":
    uvicorn.run("api:server", host="0.0.0.0", port=5000, log_level="info", reload=True)
