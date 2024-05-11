import feedparser
import requests
import datetime
import pandas as pd
import xml.etree.ElementTree as ET

import os
import logging

from datetime import timedelta, date

# 定义解析RSS源的函数
def parse_rss_feed(the_url):
    response = requests.get(url=the_url)
    feed_content = response.content
    titles, descriptions, links, categories, pub_dates = [], [], [], [], []

    # 解析XML内容
    root = ET.fromstring(feed_content)
    articles_collection = root.findall("./channel/item")

    # 提取数据并添加到相应列表
    for article in articles_collection:
        titles.append(article.find("title").text.strip())
        descriptions.append(article.find("description").text.strip())
        links.append(article.find("link").text.strip())
        pub_date_str = article.find("pubDate").text.strip()
        # 使用正确的日期时间格式来解析字符串
        pub_date = datetime.datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %Z")
        pub_dates.append(pub_date)

    # 创建DataFrame
    df = pd.DataFrame({
        "Title": titles,
        "Description": descriptions,
        "Link": links,
        "Published Date": pub_dates
    })
    df['Published Date'] = pd.to_datetime(df['Published Date'])
    df_subset = df[df['Published Date'].dt.date == (datetime.date.today() - datetime.timedelta(days=1))]
    return df_subset



# 定义解析RSS源的另一个函数
def get_feed_news(rss_url):
    feed = feedparser.parse(rss_url)
    titles, summaries, links, published_dates = [], [], [], []

    for entry in feed.entries:
        titles.append(entry.title)
        summaries.append(entry.summary)
        links.append(entry.link)
        published_dates.append(entry.published)

    df = pd.DataFrame({
        'Title': titles,
        'Summary': summaries,
        'Link': links,
        'Published Date': published_dates
    })
    df['Published Date'] = pd.to_datetime(df['Published Date'])
    df_subset = df[df['Published Date'].dt.date == (datetime.date.today() - datetime.timedelta(days=1))]
    return df_subset


# 定义将DataFrame保存到本地的函数
def save_dataframe_to_local(df: pd.DataFrame, file_path: str):
    df.to_csv(file_path, index=False)
    logging.info(f"Data saved to {file_path}")


# 定义主函数
def main():
    logging.basicConfig(level=logging.INFO)
    rss_urls = [
        'https://www.theguardian.com/international/rss',
        'https://www.theguardian.com/sport/rss',
        'https://www.theguardian.com/culture/rss',
        'https://www.theguardian.com/lifeandstyle/rss'
    ]

    # 读取RSS源并创建数据帧
    df = pd.concat([parse_rss_feed(url) for url in rss_urls], axis=0).reset_index(drop=True)

    # 确保"DaillyNews"文件夹存在
    local_path = os.path.join(os.getcwd(), "DailyNews")
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    # 创建文件名并保存数据
    file_name = f"{(datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y%m%d')}_news.csv"
    file_path = os.path.join(local_path, file_name)

    # 将数据保存到本地
    save_dataframe_to_local(df, file_path)
    print(f"File saved to {file_path}")

    # 保存第二个文件
    file_path2 = os.path.join(local_path, "today_feed.csv")
    save_dataframe_to_local(df, file_path2)
    print(f"File 2 saved to {file_path2}")


# 运行主函数
if __name__ == "__main__":
    main()
