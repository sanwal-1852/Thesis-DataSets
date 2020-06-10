import pandas as pd

# https://www.kaggle.com/grouplens/movielens-20m-dataset
df = pd.read_excel('non-avg-users.xlsx')
df = df.drop(columns=['imdbId','imdbRating'])
df['rating'] = df['rating'].div(2)
df.head(5)

# make the user ids go from 0...N-1
df.userId = df.userId - 1

# create a mapping for movie ids
unique_movie_ids = set(df.movieId.values)
movie2idx = {}
count = 0
for movie_id in unique_movie_ids:
  movie2idx[movie_id] = count
  count += 1

# add them to the data frame
# takes awhile
df['movie_idx'] = df.apply(lambda row: movie2idx[row.movieId], axis=1)


df.to_csv('edited_rating.csv', index=False)

