import json
import requests
import os
import time
import datetime
import ast

import pandas as pd
import numpy as np
from tqdm import tqdm

restaurant_data_file = './data/raw/restaurants.csv'
weather_data_file = './data/raw/weather.csv'

restaurant_df = pd.read_csv(restaurant_data_file)
restaurant_df['location.zip_code'] = restaurant_df['location.zip_code'].astype(str).str.pad(width=5, side='left',
                                                                                            fillchar='0')

weather_df = pd.read_csv(weather_data_file)
weather_df['postal_code'] = weather_df['postal_code'].astype(str).str.pad(width=5, side='left', fillchar='0')

nyt_df = pd.read_csv('./data/nytimes_covid19_data/20200405_us-counties.csv')  # always use the latest nyt source
nyt_df['fips'] = nyt_df['fips'].dropna().astype(int).astype(str)  # todo there are nan here

zip_df = pd.read_csv('./data/raw/uszips.csv')
zip_df['zip'] = zip_df['zip'].astype(str).str.pad(width=5, side='left', fillchar='0')
zip_df['county_fips_all'] = zip_df['county_fips_all'].str.split('|')
zip_df = zip_df.explode('county_fips_all')  # convert row to fip level before re-aggregate
zip_df = zip_df[['county_fips_all', 'zip']]

# restaurant
# todo: nice to haves: feature eng from categories: count of bars, cafes, etc.
restaurant_df['transactions'] = restaurant_df['transactions'].apply(lambda x: '|'.join(list(ast.literal_eval(x))))
restaurant_dummies = pd.concat([pd.get_dummies(restaurant_df['price'], drop_first=True, prefix='price'),
                                pd.get_dummies(restaurant_df['transactions'], drop_first=True, prefix='transactions'),
                                restaurant_df['location.zip_code']], axis=1)
restaurant_dummies = restaurant_dummies.groupby('location.zip_code').sum().reset_index().add_prefix('sum_')
restaurant_mean = restaurant_df.groupby('location.zip_code')[
    ['review_count', 'rating']].mean().reset_index().add_prefix('mean_')

restaurant_df = pd.merge(restaurant_mean, restaurant_dummies, left_on='mean_location.zip_code',
                         right_on='sum_location.zip_code', validate='1:1')
restaurant_df = pd.merge(restaurant_df, zip_df, left_on='sum_location.zip_code', right_on='zip', how='left')
restaurant_df = restaurant_df.groupby('county_fips_all').mean().reset_index()  # restaurant data on zip level
restaurant_df = restaurant_df.drop(restaurant_df.filter(regex='zip', axis=1).columns)

# weather
weather_df['timestamp'] = weather_df['timestamp'].str.slice(stop=10)
weather_df = pd.merge(weather_df, zip_df, left_on='postal_code', right_on='zip', how='left')
weather_df = weather_df.groupby(['county_fips_all', 'timestamp']).mean().reset_index()
weather_df = weather_df.drop(weather_df.filter(regex='zip', axis=1).columns)

# join data
df = pd.merge(nyt_df, weather_df, left_on=['fips', 'date'], right_on=['county_fips_all', 'timestamp'], how='inner')
df = pd.merge(df, restaurant_df, left_on='fips', right_on='county_fips_all', how='inner')
df.to_csv('./data/processed/time_series.csv')
