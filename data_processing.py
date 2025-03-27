import os
import pandas as pd
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

import base64
from PIL import Image
import io

# Load CSV data
def load_csv(file_path):
    df = pd.read_csv(file_path)
    return df

# Create SQL database from DataFrame
def create_sql_database(df):
    engine = create_engine(os.getenv("DATABASE_URL"))
    df.to_sql('fashion', con=engine, if_exists='replace', index=False)
    db = SQLDatabase(engine=engine)
    return db

def encode_img(uploaded_image):
    image = Image.open(uploaded_image)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

