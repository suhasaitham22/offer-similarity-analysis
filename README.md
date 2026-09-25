# Offer Similarity Analysis

A Streamlit app that finds and compares similar retail offers across brands, categories, and retailers. It started as a data science take-home style task (the in-app copy references Fetch Rewards): given messy, real-world offer datasets, build a tool where you type a brand, category, or retailer and get back the most similar offers.

## Dataset

Three CSVs, all included in the repo:

- `brand_category.csv` — brands and the product category each brand belongs to
- `categories.csv` — product categories and their parent categories
- `offer_retailer.csv` — offers with brand and retailer information

These are merged into `merged_data_fetch.csv`, which the app loads at startup.

## Approach

The app (`fetch_search.py`) has three pages:

1. **Home** — background on the task: matching offers across brands, categories, and retailers despite inconsistent naming.
2. **Data and Preprocessing** — an interactive walkthrough of the cleaning pipeline. Each step can be toggled to see its effect:
   - Fill missing retailers with "Unknown", lowercase everything, strip punctuation, remove stopwords
   - Tokenize and lemmatize with NLTK's WordNetLemmatizer
   - Join the tables: outer join of retailer and brands on `BRAND`, then joined with categories on `BRAND_BELONGS_TO_CATEGORY` = `PRODUCT_CATEGORY`, so every offer carries its brand, category, and retailer context
3. **Model** — pick TF-IDF or BERT, choose a field (Brand, Category, or Retailer), type a value, and get ranked similar offers:
   - **TF-IDF model:** `TfidfVectorizer` over the offer text, cosine similarity (`linear_kernel`) between the query and every offer
   - **BERT model:** `paraphrase-MiniLM-L6-v2` sentence embeddings with torch cosine similarity for semantic matching
   - User input is fuzzy-matched (fuzzywuzzy, ratio >= 50) before filtering, and results are ranked by cosine similarity with duplicates removed

## Results

Ranked lists of similar offers with cosine similarity scores for each query, viewable as full tables or top-5. There are no fixed accuracy metrics in the repo; it is an interactive search tool, so the "results" are the ranked offers themselves.

## Install and run

```bash
git clone https://github.com/suhasaitham22/offer-similarity-analysis.git
cd offer-similarity-analysis
pip install -r requirements.txt
streamlit run fetch_search.py
```

The app downloads NLTK data (`punkt`, `wordnet`, `stopwords`) automatically on first run.

## Tech stack

Streamlit, pandas, NLTK, scikit-learn (TF-IDF, cosine similarity), sentence-transformers, PyTorch, fuzzywuzzy.
