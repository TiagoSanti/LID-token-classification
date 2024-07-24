import requests
from bs4 import BeautifulSoup
from lxml import etree
import pandas as pd
import re


def fetch_sitemap(url):
    response = requests.get(url)
    if response.status_code == 200:
        sitemap_content = response.content
        soup = BeautifulSoup(sitemap_content, features="xml")
        return soup
    else:
        print(f"Failed to fetch sitemap: {response.status_code}")
        return None


def extract_news_metadata(soup):
    news_items = []
    for url in soup.find_all("url"):
        loc = url.find("loc").text if url.find("loc") else None
        news = url.find("news:news")
        if news:
            title = news.find("news:title").text if news.find("news:title") else None

            news_items.append(
                {
                    "url": loc,
                    "title": title,
                }
            )

            print(f"URL: {loc} | Title: {title}")
    return news_items


def clean_text(text):
    text = re.sub(r"&#[0-9]+;", "", text)
    text = " ".join(text.split())
    return text


def ends_with_punctuation(text):
    return text.endswith((".", "!", "?"))


def scrape_article(news_item):
    response = requests.get(news_item["url"])
    if response.status_code == 200:
        page_content = response.content
        tree = etree.HTML(page_content)

        content_element = tree.xpath('//article[@data-ds-component="article"]')
        if not content_element:
            print(f"No article element found for URL: {news_item['url']}")
            return None

        content_element = content_element[0]
        paragraphs = content_element.xpath(
            './/p[not(parent::div[@data-ds-component="ad"])]'
        )
        content = " ".join(
            [
                etree.tostring(para, method="text", encoding="unicode").strip()
                for para in paragraphs
            ]
        )

        title = clean_text(news_item["title"])
        content = clean_text(content)

        if not ends_with_punctuation(title):
            full_content = f"{title}. {content}"
        else:
            full_content = f"{title} {content}"

        return {
            "url": news_item["url"],
            "full_content": full_content,
        }
    else:
        print(f"Failed to fetch article: {response}")
        return None


soup = fetch_sitemap("https://www.infomoney.com.br/news-sitemap.xml")
if soup:
    news_metadata = extract_news_metadata(soup)

    articles_data = []

    for item in news_metadata:
        article_data = scrape_article(item)
        if article_data:
            articles_data.append(article_data)

    if articles_data:
        df = pd.DataFrame(articles_data)
        print(df)

        df.to_csv("datasets/infomoney_news.csv", index=False)
