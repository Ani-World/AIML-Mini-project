import sqlite3
import os

# Path to your database
DB_PATH = os.path.join(os.path.dirname(__file__), "movies.db")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Add column only if it doesn't exist
try:
    cursor.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
    print("Column added successfully.")
except sqlite3.OperationalError as e:
    if "duplicate column name" in str(e):
        print("Column already exists.")
    else:
        raise

conn.commit()
conn.close()
