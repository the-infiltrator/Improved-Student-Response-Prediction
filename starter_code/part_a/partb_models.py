from utils import *
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import CategoricalNB
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool, cv
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def train_models(X, X_val, y, y_val):
    logistic = LogisticRegressionCV(cv=10, penalty='l2', random_state=0).fit(X, y)

    gnb = CategoricalNB()
    gnb.fit(X, y)

    xgb = XGBClassifier()
    xgb.fit(np.array(X), np.array(y))

    cb = CatBoostClassifier()
    cb.fit(X, y)

    mlp = MLPClassifier(solver='lbfgs', alpha=0.01, hidden_layer_sizes=(5, 5, 5, 5), random_state=1, max_iter=1000)
    mlp.fit(X, y)

    print(f"\n \nLogistic Regression: {metrics.accuracy_score(y_val, logistic.predict(np.array(X_val)))} ")
    print(f"Categorical Naive Bayes:  {metrics.accuracy_score(y_val, gnb.predict(X_val))} ")
    print(f"Gradient Boosted Trees (XGBoost): {metrics.accuracy_score(y_val, xgb.predict(np.array(X_val)))}")
    print(f"CatBoost Algorithm: {metrics.accuracy_score(y_val, cb.predict(np.array(X_val)))}")
    print(f"Neural Network(MLP): {metrics.accuracy_score(y_val, mlp.predict(np.array(X_val)))}")


def main():
    X, X_val, y, y_val = data_prep()
    train_models(X, X_val, y, y_val)


def data_prep():
    """
    Prepare data-set for modelling
    """
    old_train_data = pd.DataFrame(load_train_csv("../data"))
    test_data = pd.DataFrame(load_public_test_csv("../data"))
    student_metadata = load_student_meta_csv("../data")
    question_metadata = load_question_meta_csv("../data")

    train_data = pd.merge(old_train_data, student_metadata, on="user_id")
    old_val_data = pd.DataFrame(load_valid_csv("../data"))
    val_data = pd.merge(old_val_data, student_metadata, on="user_id")
    X = []
    for i in range(len(train_data['user_id'])):
        X.append([train_data['user_id'][i], train_data['question_id'][i], train_data['gender'][i]])

    y = train_data['is_correct']

    X_val = []
    for i in range(len(val_data['user_id'])):
        X_val.append([val_data['user_id'][i], val_data['question_id'][i], val_data['gender'][i]])
    y_val = val_data['is_correct']

    return X, X_val, y, y_val


if __name__ == "__main__":
    # data_prep()
    main()
