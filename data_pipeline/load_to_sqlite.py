import sqlite3
import pandas as pd

def load_data():
    df = pd.read_csv("books_clean.csv")

    conn = sqlite3.connect("books.db")
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS categories(
        category_id INTEGER PRIMARY KEY,
        category_name TEXT UNIQUE
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS books(
        book_id INTEGER PRIMARY KEY,
        title TEXT,
        price_gbp REAL,
        price_inr REAL,
        rating INTEGER,
        in_stock INTEGER,
        category_id INTEGER REFERENCES categories(category_id)
    )
    """)

    categories = df["category"].unique()
    for cat in categories:
        cur.execute("INSERT OR IGNORE INTO categories(category_name) VALUES (?)", (cat,))

    for _, row in df.iterrows():
        cur.execute("SELECT category_id FROM categories WHERE category_name=?",(row["category"],))
        cat_id = cur.fetchone()[0]
        cur.execute("""
            INSERT INTO books(title, price_gbp, price_inr, rating, in_stock, category_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """,(row["title"], row["price_gbp"], row["price_inr"], row["rating"], int(row["in_stock"]), cat_id))

    conn.commit()
    conn.close()
    print("Data loaded into books.db")

if __name__ == "__main__":
    load_data()