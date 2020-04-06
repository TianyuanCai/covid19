import json
import requests

import pandas as pd
import numpy as np
import os

restaurant_file = './data/raw/restaurants.csv'


def get_restaurants(zip_code):
    """
    Download restaurant data to folder, append if file aleady exists

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

        df = df.append(tmp_df)
        n_record = min(n_record, json.loads(r.text)['total'])

    if os.path.exists(restaurant_file):
        df.to_csv(restaurant_file, index=False, header=False, mode='a')
    else:
        df.to_csv(restaurant_file, index=False)
