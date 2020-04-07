import json
import requests

import pandas as pd
import numpy as np
import os

restaurant_data_file = './data/raw/restaurants.csv'
weather_data_file = './data/raw/weather.csv'


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
    df = pd.DataFrame()

    while n_record - record_obtained > 0:
        url = f'https://api.yelp.com/v3/businesses/search?term=restaurant&location={zip_code}&limit={limit}&offset=' \
              f'{len(df)}'
        r = requests.get(url, headers={
            "Authorization": "Bearer C4f1EztmmT5nYgEqH6B4XxqIwP8Hv2xwaiXbnodamVTQ7XfnLcFBNm7pi"
                             "-2hpXagZjojbD_yyL8kPW4xqAe"
                             "-eXZqbiM8-Z4T8oz761O57z1do17P21utYvfzpXuKXnYx"})

        tmp_df = pd.json_normalize(json.loads(r.text)['businesses'])
        record_obtained += len(tmp_df)

        tmp_df = tmp_df[(tmp_df['location.zip_code'] == zip_code) & (tmp_df['is_closed'].astype(int) == 0)]
        tmp_df = tmp_df[['name', 'review_count', 'categories', 'rating', 'transactions', 'price', 'location.zip_code']]

        restaurant_df = restaurant_df.append(tmp_df)
        n_record = min(n_record, json.loads(r.text)['total'])

    if os.path.exists(restaurant_data_file):
        restaurant_df.to_csv(restaurant_data_file, index=False, header=False, mode='a')
    else:
        restaurant_df.to_csv(restaurant_data_file, index=False)


def get_weather_data(zip_code, date):
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
          f'us/history.json?period=day&timestamp_eq={date}'
    r = requests.get(url)
    weather_df = pd.json_normalize(json.loads(r.text))

    if os.path.exists(weather_data_file):
        weather_df.to_csv(weather_data_file, index=False, header=False, mode='a')
    else:
        weather_df.to_csv(weather_data_file, index=False)


if __name__ == '__main__':
    nyt_df = pd.read_csv('data/nytimes_covid19_data/20200330_us-counties.csv')

    zip_df = pd.read_csv('data/raw/uszips.csv')  # https://simplemaps.com/data/us-zips
    zip_df['zip'] = zip_df['zip'].astype(str).str.pad(width=5, side='left', fillchar='0')

    # download restaurant and weather data to raw folder
    for z in set(zip_df['zip']):
        get_restaurant_data(z)

        for d in set(nyt_df['date']):
            get_weather_data(z, d)

    # todo merge to nyt data after aggregating by zip codes
    county_zips = pd.DataFrame(zip_df.groupby(['county_name'])['zip'].apply(list)).reset_index()
    df = nyt_df.merge(county_zips, how='left', left_on='county', right_on='county_name')
