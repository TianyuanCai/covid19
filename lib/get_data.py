import json
import requests
import os
import time
import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

restaurant_data_file = './data/raw/restaurants.csv'
weather_data_file = './data/raw/weather.csv'
today = datetime.datetime.today().strftime('%Y-%m-%d')


def get_restaurant_data(zip_code):
    """
    Download restaurant data to folder, append if file aleady exists
    https://www.yelp.com/fusion

    :param zip_code: str, Example '94118'
    :return:
    """
    n_record = np.inf
    record_obtained = 0
    limit = 50  # max is 50
    restaurant_df = pd.DataFrame()

    while n_record - record_obtained > 0:
        url = f'https://api.yelp.com/v3/businesses/search?term=restaurant&location={zip_code}&limit={limit}&offset=' \
              f'{len(restaurant_df)}'
        r = requests.get(url, headers={
            "Authorization": "Bearer C4f1EztmmT5nYgEqH6B4XxqIwP8Hv2xwaiXbnodamVTQ7XfnLcFBNm7pi"
                             "-2hpXagZjojbD_yyL8kPW4xqAe"
                             "-eXZqbiM8-Z4T8oz761O57z1do17P21utYvfzpXuKXnYx"})

        try:
            tmp_df = pd.json_normalize(json.loads(r.text)['businesses'])
        except KeyError:
            return
        record_obtained += len(tmp_df)

        if len(tmp_df) > 0:
            tmp_df = tmp_df[(tmp_df['location.zip_code'] == zip_code) & (tmp_df['is_closed'].astype(int) == 0)]
            tmp_df = tmp_df[
                ['name', 'review_count', 'categories', 'rating', 'transactions', 'price', 'location.zip_code']]

            restaurant_df = restaurant_df.append(tmp_df)
            n_record = min(n_record, json.loads(r.text)['total'])
        else:
            n_record = 0

    if os.path.exists(restaurant_data_file):
        restaurant_df.to_csv(restaurant_data_file, index=False, header=False, mode='a')
    else:
        restaurant_df.to_csv(restaurant_data_file, index=False)


def get_weather_data(zip_code, date, start_date='2020-01-01'):
    """
    Download weather data to folder, append if file aleady exists
    https://developer.weathersource.com/tools/postman-collection-onpoint-api/
    covid19 Research access

    :param zip_code: str, Example '94118'
    :param date: str, Example '2020-02-01'
    :return:
    """
    api_key = '84663a2b5a93171ddb2a'
    url = f'https://api.weathersource.com/v1/{api_key}/postal_codes/{zip_code},' \
          f'us/history.json?period=day&timestamp_between={start_date},{date}'
    r = requests.get(url)
    weather_df = pd.json_normalize(json.loads(r.text))

    if os.path.exists(weather_data_file):
        weather_df.to_csv(weather_data_file, index=False, header=False, mode='a')
    else:
        weather_df.to_csv(weather_data_file, index=False)
    time.sleep(3)


if __name__ == '__main__':
    nyt_df = pd.read_csv('./data/nytimes_covid19_data/20200405_us-counties.csv')  # always use the latest nyt source
    nyt_df['fips'] = nyt_df['fips'].dropna().astype(int).astype(str)  # todo there are nan here

    zip_df = pd.read_csv('./data/raw/uszips.csv')
    zip_df['zip'] = zip_df['zip'].astype(str).str.pad(width=5, side='left', fillchar='0')
    zip_df['county_fips_all'] = zip_df['county_fips_all'].str.split('|')
    zip_df = zip_df.explode('county_fips_all')  # convert row to fip level before re-aggregate
    zip_df = zip_df[['county_fips_all', 'zip']]
    zip_df = pd.DataFrame(zip_df.groupby('county_fips_all')['zip'].apply(list)).reset_index()

    all_zips = set(zip_df['zip'])
    all_dates = set(nyt_df['date'])

    if os.path.exists(restaurant_data_file):
        restaurant_df = pd.read_csv(restaurant_data_file)
        restaurant_zips = set(restaurant_df['location.zip_code'].astype(str))
    else:
        restaurant_zips = set([])
    if os.path.exists(weather_data_file):
        weather_df = pd.read_csv(weather_data_file)

    # prioritize zipcodes in major cities
    remaining_zips = all_zips - restaurant_zips
    remaining_zips = [k for k in remaining_zips if
                      '021' in k[:3] or '001' in k[:3] or '900' in k[:3] or '941' in k[:3] or '981' in k[:3]]

    # download restaurant and weather data to raw folder
    for z in tqdm(remaining_zips):
        get_restaurant_data(z)
        get_weather_data(z, today)

    # append all zip index to nyt data frame
    df = nyt_df.merge(zip_df, how='left', left_on='fips', right_on='county_fips_all')

    # aggregate weather and restaurant data from fip to zip level

    # append
