# Content-Based Movie Recommendation System with Sentiment Analysis

A movie recommendation web application that suggests similar movies based on content and provides insights into user reviews through sentiment analysis.

The system combines a **Content-Based Recommendation Engine** using CountVectorizer and Cosine Similarity with a **Sentiment Analysis Model** using TF-IDF and Multinomial Naive Bayes. It integrates TMDB API for movie information and uses BeautifulSoup for scraping IMDb reviews.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [Recommendation Engine](#recommendation-engine)
- [Sentiment Analysis](#sentiment-analysis)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [How to Run](#how-to-run)
- [How It Works](#how-it-works)
- [Limitations and Future Improvements](#limitations-and-future-improvements)

## Overview

The Content-Based Movie Recommendation System recommends movies similar to the movie selected by the user by analysing movie metadata such as genres, directors and actors.

In addition to recommendations, the application incorporates sentiment analysis to classify IMDb user reviews as positive or negative, helping users understand audience opinions about a particular movie.

The application uses Flask as the backend framework and HTML, CSS and JavaScript for the frontend. AJAX enables asynchronous communication between the frontend and backend, while TMDB API provides movie information, posters, ratings and cast details.

## Features

- **Content-Based Recommendations:** Recommends the top 10 movies based on metadata similarity using CountVectorizer and Cosine Similarity.
- **Movie Information:** Retrieves movie descriptions, genres, ratings, release dates, runtime and posters using TMDB API.
- **Cast Information:** Displays cast details, actor biographies and other available information.
- **IMDb Review Scraping:** Extracts user reviews from IMDb using Requests and BeautifulSoup.
- **Sentiment Analysis:** Classifies individual movie reviews as positive or negative using a trained Multinomial Naive Bayes classifier.
- **Interactive Search:** Provides movie autocomplete suggestions while users type.
- **AJAX Integration:** Updates recommendations and movie information without requiring complete webpage reloads.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python 3.8 |
| Backend Framework | Flask |
| Frontend | HTML, CSS, JavaScript, jQuery, Bootstrap |
| Machine Learning | Scikit-learn |
| Recommendation Algorithm | CountVectorizer, Cosine Similarity |
| NLP | TF-IDF, Multinomial Naive Bayes |
| Web Scraping | BeautifulSoup4, Requests |
| HTML Parser | lxml |
| Data Processing | Pandas, NumPy |
| External API | TMDB API |
| Model Serialization | Pickle |
| Deployment | Gunicorn, Heroku (original deployment configuration) |

## System Architecture

The application consists of five major components:

1. **Frontend:** Provides the movie search interface, autocomplete suggestions, movie details and recommendations.
2. **Flask Backend:** Handles user requests, executes the recommendation algorithm and processes scraped reviews.
3. **Recommendation Engine:** Converts movie metadata into numerical vectors and calculates cosine similarity to identify similar movies.
4. **Sentiment Analysis Model:** Processes extracted IMDb reviews using TF-IDF and predicts their sentiment using Multinomial Naive Bayes.
5. **External Services:** TMDB API supplies movie information, while IMDb provides the reviews used for sentiment analysis.

### Application Workflow

1. The user searches for a movie through the web interface.
2. The frontend communicates with TMDB API to identify the selected movie and retrieve its information.
3. The selected movie title is sent to the Flask backend through an AJAX request.
4. The recommendation engine calculates similarity scores and returns the top 10 similar movies.
5. TMDB API provides posters and additional information for the recommended movies.
6. The application's backend uses the selected movie's IMDb ID to retrieve its review webpage.
7. BeautifulSoup extracts the available review text.
8. The saved TF-IDF vectorizer converts each review into numerical features.
9. The trained Naive Bayes classifier predicts whether each review is positive or negative.
10. The final webpage displays movie details, recommendations, cast information and review sentiment classifications.

## Recommendation Engine

The application uses Content-Based Filtering to recommend movies based on their attributes rather than user ratings or viewing histories.

### 1. Data Preprocessing

Movie metadata is collected and processed to create a combined textual representation of each movie.

The following attributes are used:

- Director name
- First three actors
- Movie genres

Missing values are handled, movie titles are normalised, and genre separators are cleaned.

These attributes are concatenated into a single feature called `comb`.

For example:

```text
Sam Worthington Zoe Saldana Sigourney Weaver
James Cameron Action Adventure Fantasy Sci-Fi
```

This combined representation captures the important metadata of a movie.

### 2. Feature Extraction using CountVectorizer

CountVectorizer converts the combined movie metadata into a numerical document-term matrix.

Each row represents a movie, and each column represents a word from the vocabulary.

The values indicate the frequency of words in the corresponding movie metadata.

```python
cv = CountVectorizer()

count_matrix = cv.fit_transform(data['comb'])
```

### 3. Similarity Calculation using Cosine Similarity

Cosine Similarity measures the similarity between two movie vectors by calculating the cosine of the angle between them.

The mathematical formula is:

\[
Similarity(A,B) = \frac{A \cdot B}{||A|| \times ||B||}
\]

The similarity scores generally range from 0 to 1 for the non-negative feature vectors used in this project.

A higher score indicates greater similarity between the movie metadata.

```python
similarity = cosine_similarity(count_matrix)
```

### 4. Generating Recommendations

Once the similarity matrix is generated:

- The selected movie is identified in the dataset.
- Its similarity scores with all other movies are retrieved.
- Movies are sorted in descending order of similarity.
- The selected movie itself is excluded.
- The top 10 similar movies are returned.

This produces personalised recommendations based on the content of the selected movie.

## Sentiment Analysis

The application integrates a machine-learning-based sentiment analysis model to classify IMDb reviews as positive or negative.

### 1. Dataset Preparation

A labelled movie-review dataset containing approximately 6,900 records is used to train the sentiment classifier.

The dataset contains:

- Review text
- Sentiment labels (1 = Positive, 0 = Negative)

### 2. Text Vectorization using TF-IDF

TF-IDF (Term Frequency–Inverse Document Frequency) converts textual reviews into numerical feature vectors.

It assigns weights to words based on their frequency within a document and their importance across the collection of documents.

Common words receive relatively lower importance, while more informative words can receive higher weights.

### 3. Model Training

The vectorized dataset is divided into training and testing subsets using an 80:20 split.

A Multinomial Naive Bayes classifier is trained to predict review sentiment.

```python
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()

clf.fit(X_train, y_train)
```

The initial held-out test evaluation achieved approximately **97.47% accuracy**.

The trained classifier and TF-IDF vectorizer are saved using Pickle for reuse in the Flask application.

### 4. Sentiment Prediction on IMDb Reviews

When a user selects a movie:

1. The movie's IMDb ID is retrieved.
2. Requests fetches the IMDb reviews webpage.
3. BeautifulSoup parses the HTML and extracts review text.
4. The saved TF-IDF vectorizer transforms each review.
5. The trained Naive Bayes classifier predicts its sentiment.
6. The application labels each review as `Good` or `Bad`.

The sentiment results are displayed alongside the corresponding reviews.

**Note:** Sentiment analysis operates independently of the recommendation algorithm. It provides additional review insights rather than influencing the movie similarity ranking.

## Data Sources

The project uses the following data sources:

- **IMDb 5000 Movie Dataset:** Initial movie metadata, including actors, directors and genres.
- **The Movies Dataset (TMDB):** Additional movie information and credits.
- **Wikipedia Movie Lists (2018–2020):** Additional movie records and cast/crew information.
- **TMDB API:** Movie information, genres, ratings, posters and cast details.
- **IMDb:** User reviews used for sentiment analysis.

## Project Structure

```text
Content-Based-Movie-Recommender/
│
├── datasets/
│   ├── data.csv
│   ├── new_data.csv
│   ├── final_data.csv
│   └── main_data.csv
│
├── static/
│   ├── recommend.js
│   ├── autocomplete.js
│   └── style.css
│
├── templates/
│   ├── home.html
│   └── recommend.html
│
├── main.py
├── main_data.csv
├── nlp_model.pkl
├── tranform.pkl
├── requirements.txt
├── Procfile
│
└── README.md
```

*Note: The exact dataset and notebook filenames may vary depending on the repository version.*

## Installation and Setup

### Prerequisites

- Python 3.8
- pip
- TMDB API key

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Navigate to the project directory:

```bash
cd Content-Based-Movie-Recommender
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate the environment:

**Windows:**

```bash
venv\Scripts\activate
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Obtain a TMDB API Key

1. Create an account on [The Movie Database](https://www.themoviedb.org/).
2. Navigate to your account settings.
3. Open the API section.
4. Apply for an API key.
5. Once approved, obtain your API key.

### 5. Configure the API Key

The original implementation uses a TMDB API key in `static/recommend.js`.

Configure the API key in the locations indicated by the existing code.

**Security recommendation:** For production use, move API requests to the backend and store the API key in an environment variable instead of exposing it in frontend JavaScript. Do not commit personal API credentials to GitHub.

## How to Run

After installing dependencies and configuring the API key, execute:

```bash
python main.py
```

Open your browser and navigate to:

```text
http://127.0.0.1:5000/
```

Search for a movie and explore its recommendations and review sentiment results.

## How It Works

The application combines two independent machine-learning pipelines:

**Movie Recommendation Pipeline**

```text
Movie Metadata
      |
      v
Data Preprocessing
      |
      v
Combined Text Features
      |
      v
CountVectorizer
      |
      v
Document-Term Matrix
      |
      v
Cosine Similarity
      |
      v
Top 10 Similar Movies
```

**Sentiment Analysis Pipeline**

```text
IMDb Reviews
      |
      v
Requests + BeautifulSoup
      |
      v
Review Text Extraction
      |
      v
TF-IDF Vectorization
      |
      v
Multinomial Naive Bayes
      |
      v
Positive / Negative Classification
      |
      v
Display Sentiment Results
```

## Limitations and Future Improvements

Potential improvements to the current implementation include:

- **Scalability:** Replace the full pairwise similarity matrix with more efficient nearest-neighbour retrieval techniques for larger movie collections.
- **Recommendation Quality:** Incorporate plot descriptions, keywords, production companies and additional metadata to improve recommendation relevance.
- **Advanced NLP:** Experiment with contextual language models such as BERT to improve sentiment classification, especially for reviews involving sarcasm and complex language.
- **Model Evaluation:** Evaluate sentiment classification using precision, recall, F1-score and confusion matrices in addition to accuracy.
- **Review Extraction:** Improve scraping reliability through robust HTML parsing, pagination, error handling and appropriate data access methods.
- **Application Performance:** Optimise API requests and cache preprocessed movie features and similarity results.
- **Recommendation Personalisation:** Explore hybrid recommendation techniques combining content-based filtering with collaborative filtering.

---

**Developed using Python, Flask, Machine Learning, NLP and TMDB API.**


