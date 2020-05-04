import datetime
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

from lib import get_data
from lib import simplified_model

rc('text', usetex=False)


def prepare_model_data(date_range, pred_day, outcome):
    df = get_data.get_model_data(date_range=date_range, pred_day=pred_day)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()  # todo check effect of imputation

    # todo feature engineering ideas
    # longitude and latitude

    df_x = simplified_model.multicollinearity_check(
        df.drop([f'day_{pred_day}_delta_cases', f'day_{pred_day}_delta_deaths', 'state', 'county', 'fips'], axis=1))

    df_x = df_x.rename({'deaths': 'past_deaths', 'cases': 'past_cases'}, axis=1)
    df_y = df[[f'day_{pred_day}_delta_{outcome}']]
    df_y = df_y.rename({f'day_{pred_day}_delta_{outcome}': outcome}, axis=1)
    df_model = pd.concat([df_x, df_y], axis=1)

    # todo add pred day End-of-Period metrics
    return df_model


model_start_time = datetime.datetime.now().strftime('%m_%d_%H_%M')  # as a flag to track separate model results

periods = 30  # total number of days to try in each period
prediction_period = 3
output_df = pd.DataFrame()

# model by start days to show changing predictability over time
for training_range in tqdm([7]):  # range of dates used for training data, from 4 days of training to 20 days
    mae = {'cases': [], 'deaths': []}  # todo outcome try actual n cases
    for day in range(periods):
        date_range = (day, day + training_range)  # interval controls the length of data collected
        y_day = day + training_range + prediction_period  # currently only predict results from n days out
        print('---', date_range, y_day)

        for y in ['cases', 'deaths']:
            # outcome = f'day_{y_day}_delta_{y}'
            tmp_df = prepare_model_data(date_range, y_day, y)

            if len(tmp_df) <= 200:
                continue

            tmp_output_df = simplified_model.linear(prepare_model_data(date_range, y_day, y),
                                                    outcome=y, family='gaussian', link='identity', seed=1,
                                                    model_name=f'{date_range[0]}_{date_range[1]}_{y_day}',
                                                    suffix=model_start_time)

            try:
                mae[y].append(list(set(tmp_output_df['mae']))[0])
                tmp_output_df['start_date'] = date_range[0]
                tmp_output_df['end_date'] = date_range[1]
                tmp_output_df['interval'] = training_range
                tmp_output_df['pred_date'] = y_day
                tmp_output_df['outcome_name'] = y
                output_df = output_df.append(tmp_output_df)
            except TypeError:
                print('--- No longer predictive')
                break

output_df.to_csv(f'reports/model_coef_{model_start_time}.csv', index=False)

# todo overall performance plot

# coefficient plots
for y in ['deaths', 'cases']:
    tmp_output_df = output_df[output_df['outcome_name'] == y]
    n_variables = len(set(tmp_output_df['names']))
    fig, axes = plt.subplots(int(np.ceil(n_variables / 4)), 4, figsize=(16, 10), sharex='all')

    i = 0
    for c in set(tmp_output_df['names']):
        axes[i // 4, i % 4].plot(tmp_output_df[tmp_output_df['names'] == c]['start_date'],
                                 tmp_output_df[tmp_output_df['names'] == c]['coefficients'])
        axes[i // 4, i % 4].set_title(f'{c}')
        axes[i // 4, i % 4].set_ylabel('Coefficient')
        axes[i // 4, i % 4].set_xlabel('Day')
        i += 1

    plt.show()

# performance plots
for metric in ['r2', 'mae']:
    performance_df = output_df[['start_date', 'r2', 'mae', 'outcome_name']].drop_duplicates()
    for y in ['deaths', 'cases']:
        tmp_performance_df = performance_df[performance_df['outcome_name'] == y]
        plt.plot(tmp_performance_df['start_date'], tmp_performance_df[metric])
        plt.title('Metric by Day')
        plt.xlabel('Day')
        plt.ylabel(metric)
        plt.show()

# try gam
# todo convert to dummies
fig = simplified_model.linear_gam(prepare_model_data((0, 7), 14, 'deaths'), 'deaths', seed=1)
plt.show()
