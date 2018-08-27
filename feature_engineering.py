import pandas as pd
import numpy as np


def build_y(df, delta_days='30 days'):
    '''build y label (bool) for each data point based on the delta_days used in the defination of churn.
    
    Arguments:
        df {[pandas dataframe]} -- input dataframe
    
    Keyword Arguments:
        delta_days {str} -- [the variable used in the defination of user's churning] (default: {'30 days'})
    
    Returns:
        y{pandas series} -- bool series created based on the defination of churn
    '''

    today = df['last_trip_date'].max()
    delta = pd.Timedelta(delta_days)
    df['churn?'] = (df['last_trip_date'] < (today-delta)) *1
    y = df['churn?']
    return y

def fill_cont_nans(df, col_list=['avg_rating_by_driver', 'avg_rating_of_driver'], grouper=['city', 'luxury_car_user']):
    '''Fill nans of continuous features

    Arguments:
        df {[pandas dataframe]} -- input dataframe

    Keyword Arguments:
        col_list {list} -- [column names of continuous features] (default: {['avg_rating_by_driver', 'avg_rating_of_driver']})
        grouper {list} -- [features to group by] (default: {['city', 'luxury_car_user']})

    Returns:
        df {[pandas dataframe]} -- output dataframe
    '''

    for col in col_list:
        df[col].fillna(df.groupby(grouper)[col].transform('median'), inplace=True)
    return df

def fill_categ_nans(df, col_list=['phone']):
    '''Fill nans of categorical features.

   Arguments:
        df {[pandas dataframe]} -- input dataframe

    Keyword Arguments:
        col_list {list} -- [column names of categorical features] (default: {['phone']})

    Returns:
        df {[pandas dataframe]} -- output dataframe
    '''

    for col in col_list:
        value = df[col].mode().values.flatten()
        # df.loc[df['phone'].isnull(), 'phone'] = value[0]
        df.loc[df[col].isnull(), col] = value[0]
    return df


def dummify(df, col_list=['city', 'phone', 'luxury_car_user']):
    '''
   Arguments:
        df {[pandas dataframe]} -- input dataframe

    Keyword Arguments:
        col_list {list} -- [column names of features to be dummified] (default: {['city', 'phone', 'luxury_car_user']})

    Returns:
        df {[pandas dataframe]} -- output dataframe
    '''

    for col in col_list:
        dummies = pd.get_dummies(df[col],prefix=col)
        df[dummies.columns] = dummies
    return df

def logify(df, col_list=['avg_dist', 'avg_rating_by_driver', 'avg_rating_of_driver']):
    '''log transform features
    Arguments:
        df {[pandas dataframe]} -- input dataframe

    Keyword Arguments:
        col_list {list} -- [features to be logrified] (default: {['avg_dist', 'avg_rating_by_driver', 'avg_rating_of_driver']})

    Returns:
        df {[pandas dataframe]} -- output dataframe
    '''

    for col in col_list:
        df[col+'_log'] = np.log(df[col]+1)
    return df

def feature_creation(df, delta_days):
    '''create the feature of user_lifespan
    
    Arguments:
        df {[pandas dataframe]} -- input dataframe
        delta_days {str} -- [the variable used in the defination of user's churning] (default: {'30 days'})

    Returns:
        df {[pandas dataframe]} -- output dataframe
    '''

    # Create user lifespan
    today = df['last_trip_date'].max()
    delta = pd.Timedelta(delta_days)
    df['user_lifespan'] = ((today-delta)-df['signup_date']).dt.days
    # Create dummy for if a user rated a driver or not
    df['user_rated_driver'] = (df['avg_rating_of_driver'].isnull() == 0) *1
    return df

def interactify(df, interacter1=['user_rated_driver'], interacter2=['avg_rating_of_driver']):
    '''create new features as interaction between two features
    Arguments:
        df {[pandas dataframe]} -- input dataframe

    Keyword Arguments:
        interacter1 {list} -- [list of column names] (default: {['user_rated_driver']})
        interacter2 {list} -- [list of column names] (default: {['avg_rating_of_driver']})

    Returns:
        df {[pandas dataframe]} -- output dataframe
    '''
    for col1, col2 in zip(interacter1, interacter2):
        df[col1+'_'+col2] = df[col1] * df[col2]
    return df


def build_X(df, model='GradientBoostingRegressor', delta_days='30 days'):
    '''feature engineer pipeline to call different functions
    
    Arguments:
        df {[pandas dataframe]} -- input dataframe
 
    Keyword Arguments:
        model {str} -- [name of sklearn model] (default: {'GradientBoostingRegressor'})
        delta_days {str} -- [the variable used in the defination of user's churning] (default: {'30 days'})
    
    Returns:
        df {[pandas dataframe]} -- output dataframe
    '''

    data = fill_categ_nans(data, ['phone'])
    data = fill_cont_nans(df, ['avg_rating_by_driver', 'avg_rating_of_driver'], ['city', 'luxury_car_user'])
    data = dummify(data, ['city', 'phone', 'luxury_car_user'])
    logify(df, ['avg_dist', 'avg_rating_by_driver', 'avg_rating_of_driver'])
    data = feature_creation(data, delta_days)
    data = interactify(data, interacter1=['user_rated_driver'], interacter2=['avg_rating_of_driver'])
    
    cols_to_keep = ['avg_dist_log', 'avg_rating_by_driver_log', 'avg_rating_of_driver_log', 'avg_surge',
       'surge_pct',
       'trips_in_first_30_days', 'weekday_pct',
        "city_King's Landing",
       'city_Winterfell', 'phone_iPhone',
        'luxury_car_user_True', 'user_lifespan', 'user_rated_driver',
       'user_rated_driver_avg_rating_of_driver']
    cols_to_keep_nonparam = ['city_Astapor', 'phone_Android',
                        'luxury_car_user_False',
                        'avg_dist', 'avg_rating_by_driver', 'avg_rating_of_driver']

    if model == 'logisticModel':
        X = data[cols_to_keep]
    else:
        X = data[cols_to_keep+cols_to_keep_nonparam]
    return X
