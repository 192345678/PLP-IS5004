import streamlit as st
import pandas as pd
import numpy as np
import crawl_news
import os
import base64
import json
import datetime
import Stock
from openai import OpenAI
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import expit
import matplotlib.pyplot as plt


MODEL = f"cardiffnlp/tweet-topic-21-multi"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


client = OpenAI()

def topic_model(input_text, model):
    print("predict")
    class_mapping = model.config.id2label
    # text = "It is great to see athletes promoting awareness for climate change."
    tokens = tokenizer(input_text, return_tensors='pt')
    output = model(**tokens)
    scores = output[0][0].detach().numpy()
    scores = expit(scores)
    predicted_class = np.argmax(scores)
    print(f"Predicted class: {class_mapping[predicted_class]}")
    return class_mapping[predicted_class]


def get_news(text, client):
    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {
          "role": "system",
          "content": "你是一个新闻专业的工作者，同时也非常熟悉前端网络编程。你需要将网上爬取的新闻数据，整理成一篇英文新闻介绍。新闻数据存储在javascript数据中， 下面是一个待处理的新闻数据的格式。请确保输出是一段完整的英文新闻。\n\"\"\"\n<p>Trump’s demeanor suggests sour mood after Stormy Daniels’s testimony, and news his former lawyer may testify next week promises more angst</p><p>Donald Trump arrived at the courtroom for his hush-money criminal trial on Friday, with apparent frustration, after sitting through two days of testimony from the adult film actor Stormy Daniels, who provided a detailed account of an alleged sexual liaison with him some 20 years ago.</p>"
        },
        {
          "role": "user",
          "content": text
        }

      ],
      temperature=0.41,
      max_tokens=1066,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    return response.choices[0].message.content


@st.cache_data()
def get_base64_of_bin_file(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def build_markup_for_logo(png_file,
    background_position="50% 10%",
    margin_top="5%",
    image_width="80%",
    image_height=""):
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

summary_file_name = f"{(datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y%m%d')}_summary.csv"
summary_file_path = os.path.join(news_path, summary_file_name)


# check if there is  a file for today's news 2024XXXX_news.csv
if not os.path.exists(file_path):
    crawl_news.main()
    df = pd.read_csv(file_path)


# check if there is a file for today's news summary 2024XXXX_summary.csv
if not os.path.exists(summary_file_path):
    Stock.main()
    df_summary = pd.read_csv(summary_file_path)

topic_file_name = f"{(datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y%m%d')}_topic.csv"
topic_file_path = os.path.join(news_path, topic_file_name)
if not os.path.exists(topic_file_path):
    df_summary = pd.read_csv(summary_file_path)
    df_summary["topic"] = df_summary["Summary"].apply(lambda text: topic_model(text, model))
    df_summary.to_csv(topic_file_path)


# show result
df_summary = pd.read_csv(summary_file_path)
df_topic = pd.read_csv(topic_file_path)
st.title("Daily News Update")
st.header("Top Trend")
# st.dataframe(df_topic)
#
# st.title('每日新闻更新')
df_topic = df_topic[:20]
# if "news_processed" not in df.columns:
#     print("start processing")
#     df["news_processed"] = df["Description"].apply(lambda desc: get_news(desc, client))
#     df.to_csv(file_path, index=False)
#     print("end processing")
# if "news_processed" in df.columns:
    # 对于数据集中的每一条新闻，创建一个超链接按钮
for index, row in df_topic.iterrows():
    # 使用st.markdown来创建超链接按钮
    # st.markdown(f"[{row['Title']}]({row['Link']})")
    # st.markdown(f"<a href='{row['Link']}' target='_blank' style='font-size: 24px;'>{row['Title']}</a>",
    #             unsafe_allow_html=True)
    # # st.write(f"主题: {row['topic']}")  # 显示新闻主题
    # st.markdown(
    #     f"<span style='background-color: blue; color: white; padding: 2px 8px; border-radius: 4px;'>Topic: {row['topic']}</span>",
    #     unsafe_allow_html=True)
    # st.write(row['Summary'])  # 显示新闻描述

    # 使用st.markdown来创建超链接按钮，并添加CSS样式来调整字体大小
    st.markdown(f"<a href='{row['Link']}' target='_blank' style='font-size: 24px;'>{row['Title']}</a>", unsafe_allow_html=True)
    # 创建一个带有颜色的标签来表示主题
    st.markdown(f"<span style='background-color: #FF5733; color: white; padding: 2px 8px; border-radius: 4px; font-size: 18px;'>Topic: {row['topic']}</span>", unsafe_allow_html=True)
    st.write(row['Summary'])  # 显示新闻摘要

