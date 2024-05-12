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

MODEL = f"cardiffnlp/tweet-topic-21-multi"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

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
    class_mapping[predicted_class] = class_mapping[predicted_class].replace('_', ' ')
    return class_mapping[predicted_class]



add_logo("image/icon.png")
news_path = os.path.join(os.getcwd(), "DailyNews")

sent_file_path = os.path.join(news_path, "news_sentiments_2024-05-05.csv")
df = pd.read_csv(sent_file_path)

df["topic"] = df["Description"].apply(lambda text: topic_model(str(text), model))
df.to_csv(sent_file_path, index=False)



# 计算每个主题的平均情感评分
average_scores = df.groupby('topic')['Sentiment'].mean().reset_index()

# # 在 Streamlit 中创建一个表格来展示DataFrame
# st.dataframe(df)

# 添加一个描述性的标题
st.title("Sentiment Analysis of News Articles")


# 计算每个主题的平均情感评分
# 计算每个主题的平均情感评分
average_scores = df.groupby('topic')['Sentiment'].mean().reset_index()

# 定义一个颜色列表
colors = ["#F7CC9B", "#B6B884", "#7CA07D", "#4F8479", "#36666E", "#2F4858" ]

# 绘制条形图来展示所有主题的平均情感评分
fig, ax = plt.subplots(figsize=(10, 6))
for i, topic in enumerate(average_scores['topic']):
    ax.bar(topic, average_scores.loc[i, 'Sentiment'], color=colors[i % len(colors)])

ax.set_title('Sentiment ANalysis by Topic')
ax.set_xlabel('Topic')
ax.set_ylabel('Average Sentiment (0-1)')
ax.set_ylim(0, 1)
ax.set_xticklabels(average_scores['topic'], rotation=45)
# 在Streamlit中展示图表
st.pyplot(fig)




# 创建一个侧边栏来选择主题
selected_topic = st.sidebar.selectbox('select a topic', average_scores['topic'].unique())

df_show = df[df['topic'] == selected_topic].sort_values(by='Sentiment', ascending=False)
df_cut = df_show[:5]
# 创建一个表格来展示平均情感评分
st.header(f"Top 5 News for topic {selected_topic}")
   # 对于数据集中的每一条新闻，创建一个超链接按钮
for index, row in df_cut.iterrows():
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
    st.markdown(f"<a href='{row['URL']}' target='_blank' style='font-size: 24px;'>{row['Title']}</a>", unsafe_allow_html=True)
    # 创建一个带有颜色的标签来表示主题
    # st.markdown(f"<span style='background-color: #FF5733; color: white; padding: 2px 8px; border-radius: 4px; font-size: 18px;'>Topic: {row['topic']}</span>", unsafe_allow_html=True)


# 绘制直方图来展示选定主题的Sentiment分数分布
fig2,  ax = plt.subplots(figsize=(10, 6))
plt.hist(df_show['Sentiment'], bins=10, color='skyblue', edgecolor='black')
plt.title(f'{selected_topic} Sentiment Scores')
plt.xlabel('Sentiment scores')
plt.ylabel('Number of Articles')
plt.grid(axis='y')
st.pyplot(fig2)


df_company = pd.read_csv("companiesofinterest.csv")
df_company.columns = ['Company']
# 创建一个下拉菜单，用于选择要查看的公司
company_name = ["Meta", "Tesla", "Grab", "Microsoft", "GE",  "Uber"]
selected_company = st.sidebar.selectbox('Select a company', df_company["Company"])
print(selected_company)
# 根据用户选择的公司，显示相关新闻

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df_filtered = df[df["Description"].notna()]
articles = df_filtered['Description'].tolist()

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(articles + [selected_company])

# 计算目标文章与其他文章的余弦相似度
similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
# 找到最相似的前20篇文章
top_20_similar_articles = np.argsort(similarities)[-10:][0]

print(top_20_similar_articles)


st.header("Company Wise Sentiment Analysis")
top_20_articles = [articles[i] for i in top_20_similar_articles[20:]]
df_article = df_filtered[df_filtered["Description"].isin(top_20_articles)]
fig3,  ax2 = plt.subplots(figsize=(10, 6))
plt.hist(df_article['Sentiment'], bins=5, color='red', edgecolor='black')
plt.title(f'Sentiment Distribution for Company: {selected_company}')
plt.xlabel('Sentiment scores')
plt.ylabel('Number of Articles')
plt.grid(axis='y')
st.pyplot(fig3)