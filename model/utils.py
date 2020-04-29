import pandas as pd
import numpy as np


def get_model_data(df, date_range=(0, 14), pred_day=21):

    # ensure full coverage of interested dates
    df = df[df['fips'].isin(df[df['days_since_10_cases'] == pred_day]['fips'].drop_duplicates())]

    # filter for training and testing dates
    df_x = df[df['days_since_10_cases'].between(date_range[0], date_range[1])]
    df_x = df_x.groupby(['state', 'county', 'fips']).mean().reset_index()  # todo play with granularity
    df_x = df_x.drop(['days_since_10_cases'], axis=1)

    # get change rate since last training date
    df_y = df[df['days_since_10_cases'].isin([date_range[1], pred_day])][
        ['fips', 'cases', 'deaths', 'days_since_10_cases']]
    df_y = df_y.sort_values(['fips', 'days_since_10_cases']).reset_index(drop=True)
    day_idx = np.where(df_y['days_since_10_cases'] == pred_day)[0]
    df_y_delta = df_y.iloc[day_idx, :].reset_index().subtract(df_y.iloc[day_idx - 1, :].reset_index())
    df_y_delta = df_y_delta.drop(['index', 'days_since_10_cases'], axis=1)
    df_y_delta['fips'] = df_y.loc[day_idx, 'fips'].reset_index(drop=True)
    df_y_delta = df_y_delta.add_prefix(f'day_{pred_day}_delta_')
    df_x = df_x.merge(df_y_delta, left_on='fips', right_on=f'day_{pred_day}_delta_fips', how='inner')
    
    return df_x