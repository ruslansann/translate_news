import sqlite3
import requests
from bs4 import BeautifulSoup
from newsapi import NewsApiClient
import logging
import config

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='parser.log',
                    filemode='a')
logger = logging.getLogger()


class Parser:
    """Парсер для извлечения контента из статьи"""

    @staticmethod
    def parse_article(url):
        try:
            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            article = soup.find("article")

            if article:
                content = " ".join([p.text for p in article.find_all("p")])
                return content
            else:
                logger.warning(f"Article not found at {url}")
                return None

        except requests.RequestException as e:
            logger.warning(f"Error while fetching article from {url}: {e}")
            return None


class NewsAPIClientWrapper:
    """Обертка для работы с NewsAPI"""

    def __init__(self, api_key):
        self.client = NewsApiClient(api_key=api_key)

    def fetch_articles(self):
        try:
            all_articles = self.client.get_everything(domains='bbc.com',
                                                      language='en',
                                                      sort_by='publishedAt')
            return all_articles.get("articles", [])
        except Exception as e:
            logger.error(f"Error fetching news articles: {e}")
            return []


class DatabaseHandler:
    """Класс для работы с SQLite базой данных"""

    def __init__(self, db_path='./DB/news.db'):
        self.db_path = db_path

    def insert_article(self, data):
        try:
            with sqlite3.connect(self.db_path) as db:
                cursor = db.cursor()
                cursor.execute(
                    """
                    INSERT INTO articles (date, title, description, content, author, url_artic, url_logo)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, data)
                db.commit()

        except sqlite3.Error as e:
            logger.error(f"Error while inserting into database: {e}")


class NewsArticleProcessor:
    """Обрабатывает статьи: парсит и сохраняет в базу данных"""

    def __init__(self, api_key, db_path='./DB/news.db'):
        self.news_api = NewsAPIClientWrapper(api_key)
        self.db_handler = DatabaseHandler(db_path)

    def process_articles(self):
        articles = self.news_api.fetch_articles()

        for x in articles:
            content = Parser.parse_article(x.get('url'))

            if content:
                data = (x.get('publishedAt', ''), x.get('title', ''),
                        x.get('description', ''), content, x.get('author', ''),
                        x.get('url', ''), x.get('urlToImage', ''))
                self.db_handler.insert_article(data)


def main():
    processor = NewsArticleProcessor(config.token)
    processor.process_articles()


if __name__ == '__main__':
    main()
