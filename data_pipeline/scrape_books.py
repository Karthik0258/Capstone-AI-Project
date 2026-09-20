import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_URL = "http://books.toscrape.com/catalogue/category/books/"

CATEGORIES = {
    "Travel": "travel_2/index.html",
    "Mystery": "mystery_3/index.html",
    "Science": "science_22/index.html"
}

def scrape_category(category_name, category_url):
    books = []
    url = BASE_URL + category_url

    while url:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")

        for item in soup.select(".product_pod"):
            title = item.h3.a["title"]
            price = item.select_one(".price_color").text
            rating = item.p["class"][1]  # e.g. "Three"
            availability = item.select_one(".availability").text.strip()

            books.append([title, price, rating, availability, category_name])

        next_page = soup.select_one("li.next a")
        if next_page:
            url = BASE_URL + category_url.replace("index.html", next_page["href"])
        else:
            url = None

    return books

def scrape_books():
    all_books = []
    for cat_name, cat_url in CATEGORIES.items():
        all_books.extend(scrape_category(cat_name, cat_url))

    df = pd.DataFrame(all_books, columns=["title","price","star_rating","availability","category"])

    df["price_gbp"] = df["price"].str.replace("£","").astype(float)

    rating_map = {"One":1,"Two":2,"Three":3,"Four":4,"Five":5}
    df["rating"] = df["star_rating"].map(rating_map)

    df["in_stock"] = df["availability"].str.contains("In stock")

    df["rating"].fillna(df["rating"].median(), inplace=True)

    df["price_inr"] = df["price_gbp"] * 105.50

    df.to_csv("books_clean.csv", index=False)
    print(f"Scraping complete. {len(df)} rows saved to books_clean.csv")

if __name__ == "__main__":
    scrape_books()