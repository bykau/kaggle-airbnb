import pandas as pd
from features import extract_user_features
from features import extract_session_features
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.grid_search import GridSearchCV
from utils import ndcg5

if __name__ == "__main__":
    def scoring(estimator, X_test, y_test):
        return ndcg5(estimator.predict_proba(X_test), y_test)

    train = extract_user_features(pd.read_csv('../data/train_users_2.csv'))
    session = extract_session_features(pd.read_csv( '../data/sessions.csv'))
    train = pd.merge(train, session, left_on='id', right_on='user_id', how='left').drop(['user_id'], axis=1)

    le = LabelEncoder()
    labels = train['country_destination'].values
    y = le.fit_transform(labels)
    X = train.drop(['country_destination', 'id'], axis=1)
    cv = [(np.where(((train['tfa_year'] == 2014) & (train['tfa_month'] == month)).values==False)[0].tolist(), np.where(((train['tfa_year'] == 2014) & (train['tfa_month'] == month)).values==True)[0].tolist()) for month in range(1, 7)]
    clf = GridSearchCV(XGBClassifier(objective='multi:softprob', seed=0, nthread=4),
                       param_grid={'max_depth': [4, 5, 6],
                                   'n_estimators': [20, 25, 30],
                                   'subsample':[0.7, 0.75, 0.8],
                                   'colsample_bytree':[0.5, 0.6, 0.7],
                                   'learning_rate':[0.2, 0.3]},
                       n_jobs=2,
                       scoring=scoring,
                       cv=cv,
                       verbose=1)

    clf.fit(X, y)
    print clf.best_score_
    print clf.grid_scores_
