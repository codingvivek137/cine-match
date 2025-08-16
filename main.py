import time
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

movies = pd.read_csv("data/movies.csv")
print("Movies dataset shape:", movies.shape)

ratings = pd.read_csv("data/ratings.csv")
print("Ratings dataset shape:", ratings.shape)

ratings = ratings.sample(frac=0.1, random_state=42)  # 10% of data
print("Sampled ratings shape:", ratings.shape)

reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

model = SVD(n_factors=50, n_epochs=10, verbose=True)  # reduced size for speed

start_time = time.time()
model.fit(trainset)
print(f"Training time: {time.time() - start_time:.2f} seconds")

predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
print("Model RMSE:", rmse)

user_id = 1
movie_id = 1
prediction = model.predict(user_id, movie_id)
print(prediction)

def recommend_movies(user_id, n=5):
    all_movies = movies['movieId'].unique()
    rated_movies = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    movies_to_predict = [m for m in all_movies if m not in rated_movies]

    predictions = [(m, model.predict(user_id, m).est) for m in movies_to_predict]
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movie_ids = [m[0] for m in predictions[:n]]
    return movies[movies['movieId'].isin(top_movie_ids)]

print(recommend_movies(user_id, n=5))
