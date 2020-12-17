from sklearn.impute import KNNImputer

from utils import *
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import CategoricalNB
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool, cv
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
from sklearn.ensemble import BaggingClassifier,StackingClassifier


def train_models(X, X_val, y, y_val, X_t, y_t):
    # logistic = LogisticRegressionCV(cv=10, penalty='l2', random_state=0).fit(X, y)
    # create a dictionary of our models
    # CATEGORICAL NAIVE BAYES

    gnb = CategoricalNB(alpha=11)
    gnb.fit(X, y)
    print(
        f"Categorical Naive Bayes:  Validation: {metrics.accuracy_score(y_val, gnb.predict(np.array(X_val)))} Test: {metrics.accuracy_score(y_t, gnb.predict(np.array(X_t)))} ")

    gnb2 = CategoricalNB(alpha=0.5)
    gnb2.fit(X, y)
    print(f"Categorical Naive Bayes:  Validation: {metrics.accuracy_score(y_val, gnb2.predict(np.array(X_val)))} Test: {metrics.accuracy_score(y_t, gnb2.predict(np.array(X_t)))} ")

    # BEST XGB
    xgb = XGBClassifier(colsample_bytree=0.5,
                        learning_rate=0.5,
                        max_depth=10,
                        min_child_weight=1,
                        n_estimators=1400,
                        scale_pos_weight=1,
                        objective='reg:squarederror',
                        subsample=0.7,
                        gamma=0.3,
                        reg_alpha=0.1,
                        booster='gbtree',
                        gpu_id=0,verbosity=1)
    xgb.fit(np.array(X), np.array(y))
    print(
        f"\nGradient Boosted Trees (XGBoost):  Validation: {metrics.accuracy_score(y_val, xgb.predict(np.array(X_val)))} Test: {metrics.accuracy_score(y_t, xgb.predict(np.array(X_t)))} ")

    xgb2 = XGBClassifier(
        learning_rate=0.2,
        n_estimators=1200,
        max_depth=8,
        min_child_weight=3,
        gamma=0.0,
        subsample=0.7,
        colsample_bytree=0.6,
        objective='binary:logistic',
        nthread=4,
        reg_alpha=0.01,
        scale_pos_weight=1.0,
    )

    xgb2.fit(np.array(X), np.array(y))
    print(
        f"\nGradient Boosted Trees (XGBoost):  "
        f"Validation: {metrics.accuracy_score(y_val, xgb2.predict(np.array(X_val)))} Test: {metrics.accuracy_score(y_t, xgb2.predict(np.array(X_t)))} ")

    #
    # # Ensemble of Naive bayes and XGBOOST

    estimators = []
    for i in range(3):
        estimators.append((f'xgb{i}', xgb))

    for i in range(1):
        estimators.append((f'xgb2{i}', xgb2))

    for i in range(1):
        estimators.append((f'gb{i}', gnb))

    for i in range(1):
        estimators.append((f'gnb2{i}', gnb2))



    # create our voting classifier, inputting our model
    # estimators.append()
    ensemble = VotingClassifier(estimators, voting="hard", n_jobs=4,verbose=True)
    ensemble.fit(np.array(X), np.array(y))
    print(f"\nVoterBagging1:  Validation {metrics.accuracy_score(np.array(y_val), ensemble.predict(np.array(X_val)))} "
          f"Test: {metrics.accuracy_score(np.array(y_t), ensemble.predict(np.array(X_t)))} ")
    estimators2 = []
    for i in range(3):
        estimators2.append((f'xgb{i}', xgb))

    for i in range(1):
        estimators2.append((f'xgb2{i}', xgb2))

    for i in range(3):
        estimators2.append((f'gb{i}', gnb))

    for i in range(1):
        estimators2.append((f'gnb2{i}', gnb2))
    ensemble2 = VotingClassifier(estimators2, voting="hard", n_jobs=4, verbose=True)
    ensemble2.fit(np.array(X), np.array(y))
    print(f"\nVoterBagging2:  Validation {metrics.accuracy_score(np.array(y_val), ensemble2.predict(np.array(X_val)))} "
          f"Test: {metrics.accuracy_score(np.array(y_t), ensemble2.predict(np.array(X_t)))} ")

    estimators3 = []
    for i in range(3):
        estimators3.append((f'xgb{i}', xgb))

    for i in range(0):
        estimators3.append((f'xgb2{i}', xgb2))

    for i in range(3):
        estimators3.append((f'gb{i}', gnb))

    for i in range(0):
        estimators3.append((f'gnb2{i}', gnb2))
    ensemble3 = VotingClassifier(estimators3, voting="hard", n_jobs=4, verbose=True)
    ensemble3.fit(np.array(X), np.array(y))
    print(f"\nVoterBagging3:  Validation {metrics.accuracy_score(np.array(y_val), ensemble3.predict(np.array(X_val)))} "
          f"Test: {metrics.accuracy_score(np.array(y_t), ensemble3.predict(np.array(X_t)))} ")

    estimators4 = []
    for i in range(4):
        estimators4.append((f'vb{i}', ensemble))

    for i in range(1):
        estimators4.append((f'vb2{i}', ensemble2))

    for i in range(5):
        estimators4.append((f'vb3{i}', ensemble3))

    for i in range(1):
        estimators4.append((f'gnb2{i}', gnb2))

    for i in range(3):
        estimators4.append((f'gnb{i}', gnb))

    for i in range(3):
        estimators4.append((f'xgb{i}', xgb))

    for i in range(1):
        estimators4.append((f'xgb2{i}', xgb2))

    ensemble4 = VotingClassifier(estimators4, voting="hard", n_jobs=4, verbose=True)
    ensemble4.fit(np.array(X), np.array(y))
    print(f"\nMEGA BAGGING:  Validation {metrics.accuracy_score(np.array(y_val), ensemble4.predict(np.array(X_val)))} "
          f"Test: {metrics.accuracy_score(np.array(y_t), ensemble4.predict(np.array(X_t)))} ")

    estimatorsstack  = [(f'en{1}', ensemble),(f'en{2}', ensemble2),(f'en{3}', ensemble3), (f'en{4}', ensemble4)]
    clstack = StackingClassifier(estimators=estimatorsstack, final_estimator=ensemble3)

    # Stack all ensembled models to create MEGA stack
    # clstack.fit(np.array(X), np.array(y))
    # print(f"\nStacked:  Validation {metrics.accuracy_score(np.array(y_val), clstack.predict(np.array(X_val)))} "
    #       f"Test: {metrics.accuracy_score(np.array(y_t), clstack.predict(np.array(X_t)))} ")


    # update_private_data(prediction=xgb.predict(np.array(X_priv)))


def update_private_data(prediction):
    private_test = load_private_test_csv("../data")
    private_test["is_correct"] = prediction
    save_private_test_csv(private_test)
    # print(pd.DataFrame(private_test))


def data_prep():
    """
    Prepare data-set for modelling
    """
    train_data = pd.DataFrame(load_train_csv("../data"))
    test_data = pd.DataFrame(load_public_test_csv("../data"))
    val_data = pd.DataFrame(load_valid_csv("../data"))
    X = []
    for i in range(len(train_data['user_id'])):
        X.append([train_data['user_id'][i], train_data['question_id'][i]])
    y = train_data['is_correct']
    # model = LogisticRegression(solver='liblinear', random_state=0).fit(X, y)
    # print(np.array(X)[:20], np.array(y).shape[0])

    X_val = []
    for i in range(len(val_data['user_id'])):
        X_val.append([val_data['user_id'][i], val_data['question_id'][i]])
    y_val = val_data['is_correct']

    X_t = []
    for i in range(len(test_data['user_id'])):
        X_t.append([test_data['user_id'][i], test_data['question_id'][i]])
    y_t = test_data['is_correct']


    # val_accuracy = model.score(X_val, y_val)
    # print(val_accuracy)
    # old_train_data = pd.DataFrame(load_train_csv("../data"))
    # old_test_data = pd.DataFrame(load_public_test_csv("../data"))
    #
    # student_metadata = load_student_meta_csv("../data")
    # question_metadata = load_question_meta_csv("../data")
    #
    # train_data = pd.merge(old_train_data, student_metadata, on="user_id")
    # old_val_data = pd.DataFrame(load_valid_csv("../data"))
    # val_data = pd.merge(old_val_data, student_metadata, on="user_id")
    #
    #
    # old_private_test =  load_private_test_csv("../data")
    # old_private_test.pop("is_correct")
    # old_private_test = pd.DataFrame(old_private_test)
    # private_test =  pd.merge(old_private_test, student_metadata, on="user_id")
    # print(old_private_test["user_id"])
    #
    # X = []
    # for i in range(len(old_train_data['user_id'])):
    #     X.append([old_train_data['user_id'][i], old_train_data['question_id'][i]])
    #
    # y = train_data['is_correct']
    #
    # X_val = []
    # for i in range(len(old_test_data['user_id'])):
    #     X_val.append([old_test_data['user_id'][i], old_test_data['question_id'][i]])
    # y_val = old_test_data['is_correct']

    #
    # old_train_data = pd.DataFrame(load_train_csv("../data"))
    # old_test_data = pd.DataFrame(load_public_test_csv("../data"))
    # old_val_data = pd.DataFrame(load_valid_csv("../data"))
    # old_private_test =  load_private_test_csv("../data")
    # old_private_test.pop("is_correct")
    # old_private_test = pd.DataFrame(old_private_test)
    # test_data = pd.merge(old_test_data, student_metadata, on="user_id", how="left")
    # train_data = pd.merge(old_train_data, student_metadata, on="user_id", how = "left")
    # val_data = pd.merge(old_val_data, student_metadata, on="user_id", how="left")
    # X = []
    # for i in range(len(train_data['user_id'])):
    #     X.append([train_data['user_id'][i], train_data['question_id'][i],train_data['gender'][i]])
    # y = train_data['is_correct']
    #
    #
    # X_val = []
    # for i in range(len(test_data['user_id'])):
    #     X_val.append([test_data['user_id'][i], test_data['question_id'][i], test_data['gender'][i]])
    # y_val = test_data['is_correct']
    #
    # X_private_test = []
    # private_test = pd.merge(old_private_test, student_metadata, on="user_id", how="left")
    #
    # for i in range(len(old_private_test['user_id'])):
    #     X_private_test.append([private_test['user_id'][i], private_test['question_id'][i], private_test['gender'][i]])

    # print(private_test["question_id"].equals(old_private_test["question_id"]))

    return X, X_val, y, y_val, X_t, y_t


def data_prep2():
    """
    Prepare data-set for modelling
    """
    old_train_data = pd.DataFrame(load_train_csv("../data"))
    test_data = pd.DataFrame(load_public_test_csv("../data"))
    student_metadata = load_student_meta_csv("../data")
    question_metadata = load_question_meta_csv("../data")

    # Uncomment to replace missing premium status with 3
    student_metadata["premium_pupil"].fillna(0, inplace=True)

    student_metadata["data_of_birth"] = pd.to_datetime(student_metadata["data_of_birth"])
    student_metadata["age"] = student_metadata["data_of_birth"].apply(lambda x: (2020 - x.year)).abs()
    # ages = np.array(student_metadata["age"])
    #
    # print(ages[ages < 0])
    # print(student_metadata["age"])
    imputer = KNNImputer(n_neighbors=5)
    student_metadata["age"] = np.rint(imputer.fit_transform(np.array(student_metadata["age"]).reshape(-1, 1)))
    # print(student_metadata["age"])
    # print(np.rint(student_metadata["age"].mean()))
    # student_metadata["age"].fillna(np.rint(student_metadata["age"].mean()), inplace=True)

    train_data = pd.merge(old_train_data, student_metadata, on="user_id")
    # print(train_data["age"])
    old_val_data = pd.DataFrame(load_valid_csv("../data"))
    val_data = pd.merge(old_val_data, student_metadata, on="user_id")
    # print('Negatives Found:')
    # print(train_data["age"].where(train_data["age"] < 0).count())

    X = []
    for i in range(len(train_data['user_id'])):
        X.append([train_data['user_id'][i], train_data['question_id'][i],
                  train_data['gender'][i],
                  train_data['premium_pupil'][i],
                  train_data['age'][i]])

    y = train_data['is_correct']

    X_val = []
    for i in range(len(val_data['user_id'])):
        X_val.append(
            [val_data['user_id'][i], val_data['question_id'][i], val_data['gender'][i],
             val_data['premium_pupil'][i], train_data['age'][i]])
    y_val = val_data['is_correct']

    return X, X_val, y, y_val


def tune_xgboost(X, X_val, y, y_val):
    xgb = XGBClassifier(
        learning_rate=0.1,
        n_estimators=1200,
        max_depth=8,
        min_child_weight=3,
        gamma=0.00,
        subsample=0.75,
        colsample_bytree=0.6,
        objective='binary:logistic',
        nthread=4,
        reg_alpha=0.01,
        scale_pos_weight=1.0,
        seed=27
    )

    params = {
        'n_estimators': [1000, 2000, 3000, 4000, 5000, 6000],
        'min_child_weight': [2, 3, 4, 5, 6, 7, 8, 9, 10],
        'gamma': [0, 0.1, 0.01, 0.001, 0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'reg_alpha': [0.1, 0.01, 0.001],
        'scale_pos_weight': [1.5, 0.66666]
    }

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=1001)
    random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=8, scoring='accuracy', n_jobs=5,
                                       cv=skf.split(X, y), verbose=3, random_state=1001)

    # Here we go
    random_search.fit(np.array(X), np.array(y))

    print('\n All results:')
    print(random_search.cv_results_)
    print('\n Best estimator:')
    print(random_search.best_estimator_)
    print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (10, 6))
    print(random_search.best_score_ * 2 - 1)
    print('\n Best hyperparameters:')
    print(random_search.best_params_)

    print(
        f"\n Gradient Boosted Trees (XGBoost): {metrics.accuracy_score(y_val, random_search.predict(np.array(X_val)))}")


if __name__ == "__main__":
    # data_prep()
    X, X_val, y, y_val, X_t, y_t = data_prep()
    train_models(X, X_val, y, y_val, X_t, y_t)

    # # Tune hparams
    # tune_xgboost(X, X_val, y, y_val)
