# -*- coding: utf-8 -*-
"""
Applying Random Forest on average and non-average people
Finding Avg and Non-avg people in different interval
"""
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#loading dataset with cleaned imdb ratings
df = pd.read_excel('userdata.xlsx')

#Taking the abs diff between rating and imdbRating and adding new column of diff
data = df.groupby('userId').apply(lambda row: abs(row['rating'] - row['imdbRating'])) \
        .reset_index(name="diff").reindex(columns=['userId', 'diff'])
data.head()

#Counting positive and negative rating for each user.
DIFF = 1.5
result = data.groupby('userId').diff.apply(lambda x: pd.Series([(x <= DIFF).sum(), \
         (x > DIFF).sum()])).unstack().reset_index()
result.columns = ['userId','P', 'N']

#Taking the percentage of every user with respective difference between
# IMDB and original rating difference.
result['sum'] = result['P'] + result['N']
result['percentage'] = (result['P']/result['sum'])*100

#Selecting average users w.r.t to similarity criteria between 
#IMDB and original ratings.
POS_PER = 80
avg_users = result[result['percentage'] >= POS_PER]
avg_users = avg_users.userId.values.tolist()
avg_users_data = df[df.userId.isin(avg_users)].sort_values(by='userId').reset_index(drop=True)
#avg_users_data.to_excel('users-1.5.xlsx')
print(len(avg_users))

#Analysis on Non-Average Users: Selecting users which are not average.
total_users  = set(df.userId.values.tolist())
non_avg_users = list(set(total_users) - set(avg_users))

#Finding all rows of diff_users
non_avg_users_data = df[df.userId.isin(non_avg_users)] \
    .sort_values(by='userId').reset_index(drop=True)
#print(len(non_avg_users))
#non_avg_users_data.to_excel('non-avg-users.xlsx')

#Applying Decision Tree
movie_data = avg_users_data


movie_data.dtypes
# Create target object and call it y
y = movie_data['rating'].div(2).round(2)
# Create X
features = ['userId', 'movieId']
X = movie_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.30, random_state=1)
print(val_X)
# Specify Model
prediction_model = RandomForestRegressor(n_estimators=300, random_state=1)
# Fit Model
prediction_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = prediction_model.predict(val_X)
val_mse = mean_squared_error(val_predictions, val_y)
val_abs = mean_absolute_error(val_predictions, val_y)

print("Validation MSE:" , val_mse)
print("Absolute Error:", val_abs)
