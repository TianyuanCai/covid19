import json
import requests
import os
import time
import datetime
import ast

import pandas as pd
import numpy as np
from tqdm import tqdm

restaurant_data_file = '../data/raw/restaurants.csv'
weather_data_file = '../data/raw/weather.csv'
today = datetime.datetime.today().strftime('%Y-%m-%d')

# # restaurant planning
# restaurant_df['transactions'] = restaurant_df['transactions'].apply(lambda x: '|'.join(list(ast.literal_eval(x))))
# restaurant_dummies = pd.concat([pd.get_dummies(restaurant_df['price'], drop_first=True, prefix='price'),
#                                 pd.get_dummies(restaurant_df['transactions'], drop_first=True,
#                                                prefix='transactions')], axis=1)
# restaurant_mean = restaurant_df.groupby('location.zip_code')['review_count', 'rating'].mean().reset_index()
# restaurant_df = pd.concat([restaurant_mean, restaurant_dummies], axis=1)
# # feature eng from categories: count of bars, etc.
#
# # weather join by datetime and weather_df['timestamp']
# weather_df['timestamp'] = weather_df['timestamp'].str.slice(stop=10)
# # aggregate weather?
#
# # append all zip index to nyt data frame
# df = nyt_df.merge(zip_df, how='left', left_on='fips', right_on='county_fips_all')
#
# # join data