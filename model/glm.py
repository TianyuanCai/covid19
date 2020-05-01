import sys

# sys.path.append('../')

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
    df_y = df[outcome]
    df_model = pd.concat([df_x, df_y], axis=1)

    # todo add pred day End-of-Period metrics
    return df_model


# Notes on write up
"""
Purpose: is to understand what's the maximum number of days we can use to predict the y
Hypothesis: 

1. 
The longer the interval of data collected, the more difficult it is to predict the change in the next 7-day
Longer interval does not help us predict better because the situation evolves fast

2. 
The intervals needed to predict death is different from that needed for cases. Long interval data helps predict
deaths better than cases
"""

model_start_time = datetime.datetime.now().strftime('%m_%d_%H_%M')  # as a flag to track separate model results

periods = 21  # total number of days to try in each period
output_df = pd.DataFrame()

# show decreasing predictability
for interval in tqdm(range(4, 24, 4)):  # range of dates used for training data, from 4 days of training to 20
    mae = {'cases': [], 'deaths': []}  # todo outcome try actual n cases
    for day in range(periods):
        date_range = (day, day + interval)  # interval controls the length of data collected
        y_day = day + interval + 7  # currently only predict results from 7 days out
        print('---', date_range, y_day)
        for y in ['cases', 'deaths']:
            outcome = f'day_{y_day}_delta_{y}'
            tmp_df = prepare_model_data(date_range, y_day, outcome)

            if len(tmp_df) <= 200:
                continue

            tmp_output_df = simplified_model.linear(prepare_model_data(date_range, y_day, outcome),
                                                    outcome=outcome, family='gaussian', link='identity', seed=1,
                                                    model_name=f'{date_range[0]}_{date_range[1]}_{y_day}',
                                                    suffix=model_start_time)

            try:
                mae[y].append(list(set(tmp_output_df['mae']))[0])
                tmp_output_df['start_date'] = date_range[0]
                tmp_output_df['end_date'] = date_range[1]
                tmp_output_df['interval'] = interval
                tmp_output_df['pred_date'] = y_day
                tmp_output_df['outcome_name'] = y
                output_df = output_df.append(tmp_output_df)
            except TypeError:
                print('--- No longer predictive')
                break

output_df.to_csv(f'reports/model_coef_{model_start_time}.csv', index=False)

# todo output separate performance table for plotting

# todo plot

# for label in mae.keys():
#     axes[i // 4, i % 4].plot(range(len(mae[label])), mae[label], label=label)
# axes[i // 4, i % 4].set_title(f'Trained on {interval}-days of Data')
# axes[i // 4, i % 4].set_ylabel('MAE')
# axes[i // 4, i % 4].set_xlabel('Prediction Start Day')
# axes[i // 4, i % 4].legend()
# i += 1

# handles, labels = axes[(i - 1) // 4, (i - 1) % 4].get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center')
# plt.show()

# plot factors predicting cases and deaths

# plot their changing coefficients over time

# plot model's changing effects over time
