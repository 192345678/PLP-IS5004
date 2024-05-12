import os
import pandas as pd
import newsapi
from newsapi.newsapi_client import NewsApiClient
from datetime import datetime, timedelta

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Obtain an Access Key for the NewsAPI
# Init news api
NEWS_API_KEY = 'fcb66481e1bc42f99060e82de9f32269'

# The News API example
# https://newsapi.org/docs/endpoints/everything

newsapi = NewsApiClient(api_key= NEWS_API_KEY)
keywrd = 'Meta stock'
my_date = (datetime.now() - timedelta(days=7)).date()

articles = newsapi.get_everything(q = keywrd,
                                  from_param = my_date.isoformat(),
                                  to = (my_date + timedelta(days = 1)).isoformat(),
                                  language="en",
                                  sort_by="relevancy",
                                  page_size = 100)

def get_articles_sentiments(keywrd, startd, sources_list=None, show_all_articles=False):
    if isinstance(startd, str):
        my_date = datetime.fromisoformat(startd)
    else:
        my_date = startd

    if sources_list:
        articles = newsapi.get_everything(q = keywrd,
                                          from_param = my_date.isoformat(),
                                          to = (my_date + timedelta(days = 1)).isoformat(),
                                          language="en",
                                          sources = ",".join(sources_list),
                                          sort_by="relevancy",
                                          page_size = 100)
    else:
        articles = newsapi.get_everything(q = keywrd,
                                          from_param = my_date.isoformat(),
                                          to = (my_date + timedelta(days = 1)).isoformat(),
                                          language="en",
                                          sort_by="relevancy",
                                          page_size = 100)

    date_sentiments_list = []
    seen = set()

    for article in articles['articles']:
        if str(article['title']) in seen:
            continue
        else:
            seen.add(str(article['title']))
            article_content = str(article['title']) + '. ' + str(article['description'])
            sentiment = sia.polarity_scores(article_content)['compound']
            date_sentiments_list.append((sentiment, article['url'],article['title'],article['description']))

    return pd.DataFrame(date_sentiments_list, columns=['Sentiment', 'URL', 'Title', 'Description'])

# 调用函数获取与关键词 "Meta stock" 相关的新闻文章
def main():
    # 获取日期字符串，表示要获取新闻的时间范围
    dt = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    # 调用 get_articles_sentiments 函数获取与关键词 "Meta stock" 相关的新闻文章
    return_articles = get_articles_sentiments(keywrd= 'Meta stock', startd = dt, sources_list = None, show_all_articles= True)

    # 将情感得分映射到0-100的范围
    return_articles['Mapped Sentiment'] = (return_articles['Sentiment'] + 1) / 2 * 100

    # 确保保存文件的文件夹存在
    folder_path = 'DailyNews'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 将DataFrame保存为CSV文件，文件名包含当前日期
    file_name = 'news_sentiments_' + dt + '.csv'
    return_articles.to_csv(os.path.join(folder_path, file_name), index=False)
    print("summary data prepared and saved successfully.")
