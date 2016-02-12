import pandas as pd
import numpy as np

def extract_user_features(data):
    data = data.drop(['date_first_booking'], axis=1)

    dac = np.vstack(data.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
    data['dac_year'] = dac[:,0]
    data['dac_month'] = dac[:,1]
    data['dac_day'] = dac[:,2]
    data = data.drop(['date_account_created'], axis=1)

    tfa = np.vstack(data.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
    data['tfa_year'] = tfa[:,0]
    data['tfa_month'] = tfa[:,1]
    data['tfa_day'] = tfa[:,2]
    data = data.drop(['timestamp_first_active'], axis=1)

    #Age
    av = data.age.values
    data['age'] = np.where(np.logical_or(av<14, av>100), -1, av)

    #One-hot-encoding features
    ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
    for f in ohe_feats:
        df_train_dummy = pd.get_dummies(data[f], prefix=f)
        data = data.drop([f], axis=1)
        data = pd.concat((data, df_train_dummy), axis=1)

    data = data.fillna(-1)
    return data

def extract_session_features(sessions):
    # total_n_sessions
    X = sessions.groupby('user_id').size().reset_index()
    X = X.rename(columns = {0:'total_n_sessions'})

    # total_secs_elapsed
    c = sessions.groupby('user_id').secs_elapsed.sum().reset_index()
    c = c.rename(columns = {'secs_elapsed':'total_secs_elapsed'})
    X = pd.merge(X, c, on='user_id', how='left')

    # max_secs_elapsed
    c = sessions.groupby('user_id').secs_elapsed.max().reset_index()
    c = c.rename(columns = {'secs_elapsed':'max_secs_elapsed'})
    X = pd.merge(X, c, on='user_id', how='left')

    # mean_secs_elapsed
    c = sessions.groupby('user_id').secs_elapsed.mean().reset_index()
    c = c.rename(columns = {'secs_elapsed':'mean_secs_elapsed'})
    X = pd.merge(X, c, on='user_id', how='left')

    # median_secs_elapsed
    c = sessions.groupby('user_id').secs_elapsed.median().reset_index()
    c = c.rename(columns = {'secs_elapsed':'median_secs_elapsed'})
    X = pd.merge(X, c, on='user_id', how='left')

    for f in ['action', 'action_type', 'action_detail', 'device_type']:
        c = sessions.groupby(['user_id', f]).size().reset_index()
        c = c.rename(columns = {0:'counts'})
        c = c.pivot(index='user_id', columns=f, values='counts').reset_index()
        c = c.rename(columns = lambda x : 'c_' + x)
        c = c.rename(columns={'c_user_id': 'user_id'})
        X = pd.merge(X, c, how='left', on='user_id')

    sessions['action_action_type_action_detail'] = sessions['action'] + sessions['action_type'] + sessions['action_detail']
    c = sessions.groupby(['user_id', 'action_action_type_action_detail']).size().reset_index()
    c = c.rename(columns = {0:'counts'})
    c = c.pivot(index='user_id', columns='action_action_type_action_detail', values='counts').reset_index()
    X = pd.merge(X, c, how='left', on='user_id')
    return X