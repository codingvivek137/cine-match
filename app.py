import time
import pandas as pd
import streamlit as st
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load data
@st.cache_data
def load_data():
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")
    ratings = ratings.sample(frac=0.1, random_state=42)  # Sample for speed
    return movies, ratings

movies, ratings = load_data()

st.title("🎬 Movie Recommender System (SVD)")

st.write(f"Movies dataset shape: {movies.shape}")
st.write(f"Ratings dataset shape: {ratings.shape}")

# Model training
@st.cache_resource
def train_model(ratings):
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    model = SVD(n_factors=50, n_epochs=10, verbose=True)
    start_time = time.time()
    model.fit(trainset)
    training_time = time.time() - start_time

    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)

    return model, training_time, rmse

with st.spinner("Training model..."):
    model, training_time, rmse = train_model(ratings)

st.success(f"✅ Model trained in {training_time:.2f} seconds with RMSE: {rmse:.4f}")

# Recommend movies function
def recommend_movies(user_id, n=5):
    all_movies = movies['movieId'].unique()
    rated_movies = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    movies_to_predict = [m for m in all_movies if m not in rated_movies]

    predictions = [(m, model.predict(user_id, m).est) for m in movies_to_predict]
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movie_ids = [m[0] for m in predictions[:n]]
    return movies[movies['movieId'].isin(top_movie_ids)]

# User input
user_id = st.number_input("Enter User ID:", min_value=1, step=1, value=1)
n_recs = st.slider("Number of Recommendations", 1, 10, 5)

if st.button("Get Recommendations"):
    st.write(recommend_movies(user_id, n=n_recs))