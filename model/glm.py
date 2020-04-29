import datetime

import pandas as pd
import numpy as np

from lib import get_data
from lib import simplified_model

date_range = (0, 14)
pred_day = 21
outcome = f'day_{pred_day}_delta_cases'

df = get_data.get_model_data(date_range=date_range, pred_day=pred_day)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

df_x = simplified_model.multicollinearity_check(
    df.drop([f'day_{pred_day}_delta_cases', f'day_{pred_day}_delta_deaths', 'state', 'county', 'fips'], axis=1))
df_y = df[outcome]

model_start_time = datetime.datetime.now().strftime('%m_%d_%H_%M')
simplified_model.linear(pd.concat([df_x, df_y], axis=1), outcome=outcome, family='gaussian', seed=1,
                        model_name='gaussian', suffix=model_start_time)
