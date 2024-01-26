from typing import Any, List

import numpy as np
from fastapi import FastAPI, HTTPException, Body, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import os
import uvicorn
import logging
from pydantic import BaseModel, Field
from sqlalchemy import create_engine
import pandas as pd
from typing import Dict, List, Optional, Union
import json
import ssl
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from statsmodels.formula.api import ols
import statsmodels.api as sm


server = FastAPI(
    title='Voice Control API',
    description='Provide data to the Voice Control App'
)

server.add_middleware(
    CORSMiddleware,
    # allow_origins=["https://www.csu-hci-experiment.online"],
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_USER = "postgres"
DB_PASSWORD = "123"
DB_HOST = "voice_control_db"
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


if __name__ == "__main__":

    uvicorn.run("voice_control_api:server", host="0.0.0.0",
                port=5000, log_level="info", reload=True)