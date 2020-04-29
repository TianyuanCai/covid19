import datetime

import pandas as pd
import numpy as np
from lib import get_data
from lib import simplified_model
import matplotlib.pyplot as plt
from matplotlib import rc

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
Hypothesis: the longer the interval of data collected, the more difficult it is to predict the change in the next 7-day
Hypothesis:  
"""

periods = 20  # total number of days to try in each period
output_df = pd.DataFrame()
fig, axes = plt.subplots(2, 4)
model_start_time = datetime.datetime.now().strftime('%m_%d_%H_%M')  # as a flag to track separate model results

# show decreasing predictability
i = 0
for interval in range(4, 20, 2):
    mae = {'cases': [], 'deaths': []}  # todo outcome try actual n cases
    for day in range(periods):
        date_range = (day, day + interval)  # interval controls the length of data collected
        y_day = day + interval + 7  # currently only predict results from 7 days out
        print('---', date_range, y_day)
        for y in ['cases', 'deaths']:
            outcome = f'day_{y_day}_delta_{y}'
            tmp_output_df = simplified_model.linear(prepare_model_data(date_range, y_day, outcome),
                                                    outcome=outcome, family='gaussian', link='identity', seed=1,
                                                    model_name=f'{date_range[0]}_{date_range[1]}_{y_day}',
                                                    suffix=model_start_time)
            try:
                mae[y].append(list(set(tmp_output_df['mae']))[0])
                output_df = output_df.append(tmp_output_df)
            except TypeError:
                print('--- No longer predictive')
                break

    for label in mae.keys():
        axes[i % 2, i // 2].plot(range(len(mae[label])), mae[label], label=label)
    axes[i % 2, i // 2].set_title(f'Trained on {interval}-days of Data')
    axes[i % 2, i // 2].set_ylabel('MAE')
    axes[i % 2, i // 2].set_xlabel('Prediction Start Day')
    i += 1

plt.show()

# plt.title('$R^2$ over Prediction Periods')
# plt.legend()
# plt.show()

# plot factors predicting cases and deaths

# plot their changing coefficients over time

# plot model's changing effects over time
