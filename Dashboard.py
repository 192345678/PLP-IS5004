import streamlit as st
import pandas as pd
import numpy as np
import crawl_news
import os
import base64
import json
from datetime import datetime
@st.cache_data()
def get_base64_of_bin_file(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def build_markup_for_logo(
    png_file,
    background_position="50% 10%",
    margin_top="5%",
    image_width="80%",
    image_height="",
):
    binary_string = get_base64_of_bin_file(png_file)
    return """
            <style>
                [data-testid="stSidebarNav"] {
                    background-image: url("data:image/png;base64,%s");
                    background-repeat: no-repeat;
                    background-position: %s;
                    margin-top: %s;
                    background-size: %s %s;
                }
            </style>
            """ % (
        binary_string,
        background_position,
        margin_top,
        image_width,
        image_height,
    )

def add_logo(png_file):
    logo_markup = build_markup_for_logo(png_file)
    st.markdown(
        logo_markup,
        unsafe_allow_html=True,
    )

add_logo("image/icon.png")

news_path = os.path.join(os.getcwd(), "DailyNews")
file_name = f"{(datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y%m%d')}_news.csv"
file_path = os.path.join(news_path, file_name)


if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    st.title("Daily News")
    st.dataframe(df)
else:
    crawl_news.main()
    df = pd.read_csv(file_path)