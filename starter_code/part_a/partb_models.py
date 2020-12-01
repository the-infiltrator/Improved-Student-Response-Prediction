from utils import *
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import CategoricalNB
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool, cv
import pandas as pd

def train_models(train_data, val_data, test_data):

    X = []
    for i in range(len(train_data['user_id'])):
        X.append([train_data['user_id'][i], train_data['question_id'][i]])
    y = train_data['is_correct']

    logistic = LogisticRegressionCV(cv=10, penalty='l2', random_state=0).fit(X, y)

    gnb = CategoricalNB()
    nb = gnb.fit(X,y)

    xgb = XGBClassifier()
    xgb.fit(np.array(X), np.array(y))

    cb  = CatBoostClassifier()
    cb.fit(X,y)
    X_val = []
    for i in range(len(val_data['user_id'])):
        X_val.append([val_data['user_id'][i], val_data['question_id'][i]])
    y_val = val_data['is_correct']


    lgval_accuracy = logistic.score(X_val, y_val)

    print(f"Logistic Regression: {lgval_accuracy} \n"
          f"Categorical Naive Bayes:  {metrics.accuracy_score(y_val, gnb.predict(X_val))} \n"
          f"Gradient Boosted Trees (XGBoost): {metrics.accuracy_score(y_val, xgb.predict(np.array(X_val)))}\n"
          f"CatBoost Algorithm: {metrics.accuracy_score(y_val, cb.predict(np.array(X_val)))}")


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    train_models(train_data, val_data, test_data)
    student_metadata = load_student_meta_csv("../data")
    # student_metadata =
    question_metadata = load_question_meta_csv("../data")
    # print(train_data.keys())
    # print(student_metadata.keys())




def data_prep():
    """
    Prepare data-set for modelling
    """
    train_data = pd.DataFrame(load_train_csv("../data"))
    test_data = pd.DataFrame(load_public_test_csv("../data"))
    student_metadata = load_student_meta_csv("../data")
    question_metadata = load_question_meta_csv("../data")
    print(train_data)
    print(pd.merge(train_data,student_metadata,on="user_id"))


if __name__ == "__main__":
    # data_prep()
    main()