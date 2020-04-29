import datetime
import json
import os
import re
import time

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

income_path = './data/income/'
processed_path = './data/processed/'
mobility_path = './data/raw/'

restaurant_data_file = './data/raw/restaurants.csv'
weather_data_file = './data/raw/weather.csv'
nyt_data_file = './data/nytimes_covid19_data/20200423_us-counties.csv'
processed_data_file = './data/processed/time_series_all.csv'

nyt_df = pd.read_csv(nyt_data_file)
income_df = pd.read_csv(income_path + 'income.csv')
states_name_df = pd.read_csv(income_path + 'states_name.csv')
population_df = pd.read_csv(income_path + 'population.csv')
mobility_df = pd.read_csv(mobility_path + 'Global_Mobility_Report.csv', low_memory=False)

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


def get_weather_data(zip_code, end_date, start_date='2020-01-01'):
    """
    Download weather data to folder, append if file aleady exists
    https://developer.weathersource.com/tools/postman-collection-onpoint-api/
    covid19 Research access

    :param start_date:
    :param zip_code: str, Example '94118'
    :param end_date: str, Example '2020-02-01'
    :return:
    """
    api_key = '84663a2b5a93171ddb2a'
    url = f'https://api.weathersource.com/v1/{api_key}/postal_codes/{zip_code},' \
          f'us/history.json?period=day&timestamp_between={start_date},{end_date}'
    r = requests.get(url)
    try:
        weather_data = pd.json_normalize(json.loads(r.text))
    except json.decoder.JSONDecodeError:
        return

    if os.path.exists(weather_data_file):
        weather_data.to_csv(weather_data_file, index=False, header=False, mode='a')
    else:
        weather_data.to_csv(weather_data_file, index=False)


def get_zip_mapping():
    zip_data = pd.read_csv('./data/raw/uszips.csv')
    zip_data['zip'] = zip_data['zip'].astype(str).str.pad(width=5, side='left', fillchar='0')
    zip_data['county_fips_all'] = zip_data['county_fips_all'].str.split('|')
    zip_data = zip_data.explode('county_fips_all')  # convert row to fip level before re-aggregate
    zip_data = zip_data[['county_fips_all', 'zip']]
    return zip_data


def aggregate_data(weather_file=weather_data_file, nyt_file=nyt_data_file):
    weather_data = pd.read_csv(weather_file)
    weather_data['postal_code'] = weather_data['postal_code'].astype(str).str.pad(width=5, side='left', fillchar='0')

    nyt_df = pd.read_csv(nyt_data_file).dropna()
    nyt_df['fips'] = nyt_df['fips'].astype(int).astype(str).str.pad(width=5, side='left', fillchar='0')

    zip_df = get_zip_mapping()

    # weather
    weather_data['timestamp'] = weather_data['timestamp'].str.slice(stop=10)
    weather_data = pd.merge(weather_data, zip_df, left_on='postal_code', right_on='zip', how='left')
    weather_data = weather_data.groupby(['county_fips_all', 'timestamp']).mean().reset_index()
    weather_data = weather_data.drop(weather_data.filter(regex='zip', axis=1).columns)

    # join data
    agg_df = pd.merge(nyt_df, weather_data, left_on=['fips', 'date'], right_on=['county_fips_all', 'timestamp'],
                      how='inner')
    agg_df = agg_df.drop(['county_fips_all', 'timestamp'], axis=1)

    return agg_df


def get_model_data(date_range=(0, 14), pred_day=21):
    data = pd.read_csv(processed_data_file)

    # ensure full coverage of interested dates
    data = data[data['fips'].isin(data[data['days_since_10_cases'] == pred_day]['fips'].drop_duplicates())]

    # filter for training and testing dates
    data_x = data[data['days_since_10_cases'].between(date_range[0], date_range[1])]
    data_x = data_x.groupby(['state', 'county', 'fips']).mean().reset_index()  # todo play with granularity
    data_x = data_x.drop(['days_since_10_cases'], axis=1)

    # get change rate since last training date
    data_y = data[data['days_since_10_cases'].isin([date_range[1], pred_day])][
        ['fips', 'cases', 'deaths', 'days_since_10_cases']]
    data_y = data_y.sort_values(['fips', 'days_since_10_cases']).reset_index(drop=True)
    day_idx = np.where(data_y['days_since_10_cases'] == pred_day)[0]
    data_y_delta = data_y.iloc[day_idx, :].reset_index().subtract(data_y.iloc[day_idx - 1, :].reset_index())
    data_y_delta = data_y_delta.drop(['index', 'days_since_10_cases'], axis=1)
    data_y_delta['fips'] = data_y.loc[day_idx, 'fips'].reset_index(drop=True)
    data_y_delta = data_y_delta.add_prefix(f'day_{pred_day}_delta_')
    data_x = data_x.merge(data_y_delta, left_on='fips', right_on=f'day_{pred_day}_delta_fips', how='inner')

    return data_x


update_weather = False

if __name__ == '__main__':
    # get zipcode data
    zip_df = get_zip_mapping()
    all_zips = set(zip_df['zip'])

    if os.path.exists(weather_data_file):
        weather_df = pd.read_csv(weather_data_file)
        existing_zips = set(weather_df['postal_code'].astype(str))
    else:
        existing_zips = set([])

    # update weather dada
    if update_weather:
        nyt_df['fips'] = nyt_df['fips'].dropna().astype(int).astype(str)
        zip2fip = zip_df.groupby('county_fips_all').first().reset_index()  # get weather by first zip in fip
        remaining_zips = set(zip2fip['zip']) - existing_zips

        last_len = 0
        for z in tqdm(remaining_zips):
            get_weather_data(z, today)

            weather_df = pd.read_csv(weather_data_file)
            existing_zips = set(weather_df['postal_code'].astype(str))

            if len(existing_zips) - last_len == 0:  # check if hitting limits
                time.sleep(60)
            else:
                time.sleep(5)
            last_len = len(existing_zips)

    # aggregate weather data
    df = aggregate_data(weather_file=weather_data_file, nyt_file=nyt_data_file)

    # calculate relative dates to the first day with 10 cases
    first_date_df = nyt_df[nyt_df['cases'] >= 10].sort_values(['county', 'date']).groupby('fips')[
        'date'].first().reset_index()
    first_date_df['fips'] = first_date_df['fips'].astype(int).astype(str).str.pad(width=5, side='left', fillchar='0')
    first_date_df.columns = ['fips', 'first_date']
    df = df.merge(first_date_df, on='fips', how='inner')
    df['first_date'] = pd.to_datetime(df['first_date'])
    df['date'] = pd.to_datetime(df['date'])
    df['days_since_10_cases'] = (df['date'] - df['first_date']).dt.days
    df = df.drop(['first_date'], axis=1)
    df['date'] = df['date'].astype(str)

    # mobility data
    us_mobility_df = mobility_df.loc[mobility_df['country_region'] == 'United States'].reset_index()
    us_mobility_df = us_mobility_df.drop(['index', 'country_region_code', 'country_region'], axis=1)
    us_mobility_df.insert(1, 'county', us_mobility_df['sub_region_2'].str.lower())
    us_mobility_df = us_mobility_df.drop(['sub_region_2'], axis=1)
    us_mobility_df = us_mobility_df.loc[~pd.isna(us_mobility_df['county'])]
    us_mobility_df['county'] = us_mobility_df['county'].map(lambda x: re.sub(' county| city', '', x)).str.lower()
    us_mobility_df.insert(1, 'state', us_mobility_df['sub_region_1'].str.lower())
    us_mobility_df = us_mobility_df.drop(['sub_region_1'], axis=1)
    df['county'] = df['county'].str.lower()
    df['state'] = df['state'].str.lower()

    # take only subset of counties according to proccessed data
    counties = pd.unique(df['county'].str.lower().map(lambda x: re.sub(' city', '', x)))
    us_mobility_df = us_mobility_df.loc[us_mobility_df['county'].str.lower().isin(counties)]

    # difference in counties
    difference = set(pd.unique(counties)) - set(us_mobility_df['county'].str.lower().unique())
    columns = set(us_mobility_df.columns) - set(['state', 'county', 'date'])
    agg = {key: 'mean' for key in columns}
    us_mobility_grouped_df = us_mobility_df.groupby(['state', 'county', 'date']).aggregate(agg).reset_index()
    weather_mobility_df = pd.merge(df, us_mobility_grouped_df, on=['date', 'state', 'county'], how='left')

    # merge population and income data
    income_split = np.split(income_df, income_df[income_df.isnull().all(1)].index)[1:]
    income_clean = pd.DataFrame()
    for state in income_split:
        state_name = state['County'][1:2].values
        state['state'] = state_name[0]
        if state_name == 'District of Columbia':
            income_clean = income_clean.append(state[1:2])
        else:
            income_clean = income_clean.append(state[2:])

    income_clean['state'] = income_clean['state'].str.lower()
    income_clean['county'] = income_clean['County'].str.lower()
    income_clean['income_2018'] = income_clean['2,018']
    income_clean = income_clean.drop(['County', '2,018'], axis=1)
    income_clean = income_clean[['county', 'state', 'income_2018']]
    income_clean['income_2018'] = income_clean['income_2018'].map(lambda x: int(re.sub(',', '', x)))

    # find index of
    states = population_df['State']
    index_list = []
    for state in states.unique():
        index_list.append(states[(states == state)].index[0])

    population_df = population_df.drop(index_list)
    population_df['Area_Name'] = population_df['Area_Name'].map(lambda x: re.sub(' County| City', '', x)).str.lower()

    dictionary = {short: long for (short, long) in zip(states_name_df['Code'], states_name_df['State'])}
    population_df['State'] = population_df['State'].map(dictionary).str.lower()

    population_df = population_df[['State', 'Area_Name', 'POP_ESTIMATE_2018']]
    population_df.columns = ['state', 'county', 'pop_2018']
    population_df['pop_2018'] = population_df['pop_2018'].map(lambda x: int(re.sub(',', '', x)))

    # merge income
    income_combined = pd.merge(weather_mobility_df, income_clean, on=['state', 'county'])
    df = pd.merge(income_combined, population_df, on=['state', 'county'], how='left')

    df.to_csv(processed_data_file, index=False)
