# Book Recommendation Data Explorer
 
# Step 1 - Import Libraries
import nltk
import pandas as pd
import matplotlib.pyplot as plt
 
# Step 2 - Download and Load the Gutenberg Corpus
nltk.download("gutenberg")
 
from nltk.corpus import gutenberg
 
book_files = gutenberg.fileids()
 
print("Books available in the Gutenberg corpus:")
print(book_files)
 
 
# Step 3 - Structure the Book Data
book_data = []
 
for file_name in book_files:
 
    # Read all words from the book
    words = gutenberg.words(file_name)
 
    # Create a readable title
    title = (
        file_name
        .replace(".txt", "")
        .replace("-", " ")
        .title()
    )
 
    # Count total words
    total_words = len(words)
 
    # Count unique alphabetic words
    unique_words = len(
        set(
            word.lower()
            for word in words
            if word.isalpha()
        )
    )
 
    # Add one book record
    book_data.append(
        {
            "book": title,
            "total_words": total_words,
            "unique_words": unique_words
        }
    )
 
 
# Step 4 - Convert to a DataFrame
books_df = pd.DataFrame(book_data)
 
print("\nStructured Book Dataset:")
print(books_df.head())
 
 
# Step 5 - Clean the Data
print("\nMissing Values:")
print(books_df.isnull().sum())
 
books_df = books_df.drop_duplicates(
    subset="book"
)
 
 
# Step 6 - Create a New Feature
books_df["vocabulary_ratio"] = (
    books_df["unique_words"]
    / books_df["total_words"]
)
 
print("\nBook Features:")
print(books_df.head())
 
 
# Step 7 - Sort the Books
books_df = books_df.sort_values(
    "total_words",
    ascending=False
)
 
print("\nBooks Sorted by Length:")
print(books_df)
 
 
# Step 8 - Recommend Similar Books
selected_book = "Austen Emma"
 
selected_length = books_df.loc[
    books_df["book"] == selected_book,
    "total_words"
].iloc[0]
 
books_df["length_difference"] = abs(
    books_df["total_words"]
    - selected_length
)
 
recommendations = (
    books_df[
        books_df["book"] != selected_book
    ]
    .sort_values("length_difference")
    .head(5)
)
 
print(
    "\nBooks similar in length to",
    selected_book
)
 
print(
    recommendations[
        [
            "book",
            "total_words",
            "unique_words"
        ]
    ]
)
 
 
# Step 9 - Visualize Book Lengths
top_books = books_df.head(10)
 
plt.figure(figsize=(11, 6))
 
plt.bar(
    top_books["book"],
    top_books["total_words"]
)
 
plt.title("Word Count of Gutenberg Books")
plt.xlabel("Book")
plt.ylabel("Total Words")
 
plt.xticks(
    rotation=45,
    ha="right"
)
 
plt.tight_layout()
plt.show()
