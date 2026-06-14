# Veloria Tech ML Intern Assignment

**Name : Kushal Kamlikar**

---

## About This Project

So this assignment was my first time doing web scraping and machine learning together. I picked cricket data because I follow cricket and thought it would be more interesting to work with real match data than some random dataset.

I built three things:

- A scraper that pulls match data from HowStat.com
- A model that tries to predict which team wins
- A semantic search tool that lets you search matches by meaning

---

## How to Run

First install the libraries:
pip install requests beautifulsoup4 pandas scikit-learn sentence-transformers chromadb

Then run each file:
python scraper.py
python model.py
python rag_search.py

Make sure you run scraper.py first since model.py and rag_search.py both need the match_data.csv file it creates.

---

## Task 1 — Scraping

I scraped from HowStat.com since it had clean HTML and was easy to parse. The script collects the last 10 completed ODI matches including date, both teams, venue, result and winner.

One problem I ran into was duplicate rows in the HTML — the same match was appearing multiple times. I fixed this by keeping a set of match keys and skipping any row I had already seen.

Another issue was team names like "2025-2026 Sri Lanka v. England" — I had to split and clean the string carefully to get just the team names.

Output is saved as match_data.csv.

---

## Task 2 — ML Model

I used Logistic Regression because it is the simplest algorithm and a good starting point for small datasets.

Features I gave the model:

- Team 1 (encoded as number)
- Team 2 (encoded as number)
- Venue (encoded as number)

Target: Winner (encoded as number)

Results:

- Accuracy: 0.00%
- F1 Score: 0.00

Honestly the accuracy is poor and I expected that. We only have 9 rows of data after cleaning which is nowhere near enough for a model to learn real patterns. If I had 1000+ matches the accuracy would be much better. I still built the full pipeline correctly — data loading, encoding, splitting, training, predicting and evaluating.

---

## Task 3 — Semantic Search

This was the most interesting part. I used the sentence-transformers library to convert each match into a vector of 384 numbers that represents its meaning. These vectors are stored in ChromaDB.

When you search for something like "away team won" — it converts your query to a vector too and finds the 3 most similar matches based on meaning, not exact words.

For example searching "England won the match" correctly returns the matches where England won even though the sentences are worded differently.

---

## Libraries Used

- requests — to download the webpage
- beautifulsoup4 — to read and parse the HTML
- pandas — to work with the data as a table
- scikit-learn — for the ML model and evaluation metrics
- sentence-transformers — to convert sentences into vectors
- chromadb — to store and search vectors
