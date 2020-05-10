import datetime
import h2o
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.ensemble
import sklearn.ensemble
from h2o.grid.grid_search import H2OGridSearch
from joblib import Parallel, delayed
from pygam import pygam
from sklearn.metrics import roc_curve, auc
from statsmodels.stats.outliers_influence import variance_inflation_factor

h2o.init(max_mem_size="16G")  # specify max number of bytes. uses all cores by default.
h2o.remove_all()  # clean slate, in case cluster was already running

from h2o.estimators.glm import H2OGeneralizedLinearEstimator

matplotlib.rc('text', usetex=True)


def multicollinearity_check(exog, thresh=5.0, frac=1):
    """
    Recursive VIF to ensure
    1. The variable with the largest VIF is dropped first
    2. Only the remaining columns are used for prediction
    """
    variables = [exog.columns[i] for i in range(exog.shape[1])]
    exog_subset = exog.sample(frac=frac)
    while True:
        # run VIF in parallel
        vif = Parallel(n_jobs=4, temp_folder='/tmp')(
            delayed(variance_inflation_factor)(exog_subset[variables].values, ix) for ix in range(len(variables)))
        max_loc = vif.index(max(vif))
        if max(vif) > thresh:
            variables.pop(max_loc)
        else:
            break

    return exog[[i for i in variables]]


def linear(data, outcome, family, link='identity', seed=1, model_name='', suffix=''):
    """
    Fit linear model using H2O

    we first use L1 regularization on the subset of variables to do further feature selection based on the standardized
    and normalized data set. Using the subset of the variable selected by regularization, we then fit an unregularized
    and unstandardized model in order to retain the interpretability of the model. All these processes are performed in
    the helper function folded into the simplified_model module.

    :param data:
    :param outcome:
    :param family:
    :param link:
    :param seed:
    :param model_name:
    :param suffix:
    :return:
    """
    data = pd.concat([data.drop(outcome, axis=1), data[outcome]], axis=1)
    df = h2o.H2OFrame(data)

    train, valid = df.split_frame([0.7], seed=seed)
    x = df.col_names[:-1]
    y = df.col_names[-1]

    hyper_params = {'lambda': list(np.logspace(-3, 3, 30))}
    search_criteria = {'strategy': 'RandomDiscrete',
                       'max_models': 30,
                       'stopping_metric': 'MAE',
                       'stopping_tolerance': 0.0001,
                       'stopping_rounds': 5,
                       'seed': seed}

    glm_grid_lasso = H2OGridSearch(
        model=H2OGeneralizedLinearEstimator(family=family, link=link, nfolds=5,
                                            standardize=True,
                                            remove_collinear_columns=False),
        hyper_params=hyper_params, search_criteria=search_criteria)

    try:
        glm_grid_lasso.train(x=x, y=y, training_frame=train, validation_frame=valid)
    except TypeError:
        print('All columns are collinear')
        return

    glm_lasso_grid_perf = glm_grid_lasso.get_grid(sort_by='MAE', decreasing=False)

    # check if lasso includes variables
    best_mae = h2o.get_model(glm_lasso_grid_perf.model_ids[0]).mae(valid=True)
    for idx in glm_lasso_grid_perf.model_ids:
        best_lasso = h2o.get_model(idx)
        subset_vars = [k for k, v in best_lasso.coef().items() if np.abs(v) >= 0.001][1:]
        delta_mae = h2o.get_model(idx).mae(valid=True) - best_mae  # calculate diff w leader
        if len(subset_vars) > 0:
            break

    # refit on selected variables
    # todo update
    subset_vars = list(set([x.split('.')[0] for x in subset_vars]))
    subset_df = df[:, subset_vars + [outcome]]
    subset_x = subset_df.col_names[:-1]
    subset_y = subset_df.col_names[-1]
    train, valid = subset_df.split_frame([0.7], seed=seed)

    try:
        glm_unreg = H2OGeneralizedLinearEstimator(model_id='glm_v1_unpenalized', family=family,
                                                  link=link,
                                                  compute_p_values=True,
                                                  lambda_=0,
                                                  standardize=False,
                                                  remove_collinear_columns=True)

        glm_unreg.train(x=subset_x, y=subset_y, training_frame=train, validation_frame=valid)
        glm_unreg._model_json['output']['coefficients_table'].as_data_frame()
    except:
        print(model_name)
        print('No single variable significantly predict the outcome more than others')
        return

    coeff_table = glm_unreg._model_json['output']['coefficients_table'].as_data_frame()[
        ['names', 'coefficients', 'p_value']]
    var_imp_pd_lasso = pd.DataFrame(glm_unreg.varimp(True))[['variable', 'percentage']]
    feature_df = pd.merge(coeff_table, var_imp_pd_lasso, left_on='names', right_on='variable')

    # todo proportion needs update for categorical variables
    # add variable usage proportion
    prop_df = pd.DataFrame()
    for v in feature_df['variable']:
        if v in data.columns:
            prop_df = prop_df.append(
                pd.DataFrame([[v, (data[v] > 0).sum() / len(data)]], columns=['variable', 'proportion']),
                ignore_index=True)
        else:
            prop_df = prop_df.append(pd.DataFrame([[v, (data[v.split('.')[0]] == v.split('.')[1]).sum() / len(data)]],
                                                  columns=['variable', 'proportion']), ignore_index=True)

    prop_df.columns = ['variable', 'proportion']
    feature_df = pd.merge(feature_df, prop_df, on='variable')

    feature_df['variable_category'] = [x.split('_')[0].title() for x in feature_df['variable']]
    feature_df['variable'] = [' '.join(x.split('_')[1:]).title() for x in feature_df['variable']]
    feature_df['time'] = datetime.datetime.now()
    feature_df['family'] = family
    feature_df['link'] = link
    feature_df['model'] = model_name
    feature_df['outcome'] = outcome.replace('_', ' ').title()
    feature_df['obs'] = subset_df.shape[0]
    feature_df['input_n_features'] = df.shape[1]
    feature_df['reg_n_features'] = subset_df.shape[1]
    feature_df['mse'] = glm_unreg.mse(valid=True)
    feature_df['mae'] = glm_unreg.mae(valid=True)
    feature_df['aic'] = glm_unreg.aic(valid=True)
    feature_df['r2'] = glm_unreg.r2(valid=True)
    feature_df['RMSE'] = glm_unreg.rmse(valid=True)
    feature_df['delta_mae_lasso'] = delta_mae

    # model_coef_file = '../reports/model_coef_{}.csv'.format(suffix)
    # if os.path.exists(model_coef_file):
    #     feature_df.to_csv(model_coef_file, index=False, header=False, mode='a')
    # else:
    #     feature_df.to_csv(model_coef_file, index=False)

    return feature_df