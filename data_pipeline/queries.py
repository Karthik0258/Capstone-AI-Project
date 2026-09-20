import sqlite3
import pandas as pd

def run_queries():
    conn = sqlite3.connect("books.db")
    cur = conn.cursor()

    q1 = "SELECT title, price_gbp FROM books WHERE rating=5"
    print("Query 1:", cur.execute(q1).fetchall())

    q2 = "SELECT title, price_inr FROM books ORDER BY price_inr DESC LIMIT 10"
    print("Query 2:", cur.execute(q2).fetchall())

    q3 = "SELECT DISTINCT rating FROM books"
    print("Query 3:", cur.execute(q3).fetchall())

    q4 = "SELECT title, price_gbp FROM books WHERE rating IN (4,5)"
    print("Query 4:", cur.execute(q4).fetchall())

    q5 = """
    SELECT b.title, c.category_name, b.rating
    FROM books b
    JOIN categories c ON b.category_id=c.category_id
    ORDER BY b.rating DESC
    LIMIT 10
    """
    print("Query 5:", cur.execute(q5).fetchall())

    df_top = pd.read_sql(q1, conn)
    print("\nPandas DataFrame from Query 1:\n", df_top.head())

    books_df = pd.read_sql("SELECT * FROM books", conn)
    cats_df = pd.read_sql("SELECT * FROM categories", conn)
    merged = pd.merge(books_df, cats_df, on="category_id")
    print("\nPandas merge equivalent of JOIN:\n", merged[["title","category_name","rating"]].sort_values("rating",ascending=False).head(10))

    conn.close()

if __name__ == "__main__":
    run_queries()