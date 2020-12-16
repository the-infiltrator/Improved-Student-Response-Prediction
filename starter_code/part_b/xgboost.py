from xgboost import XGBClassifier
import numpy as np
from starter_code.utils import *


def train_models(X, X_val, X_test, y, y_val, y_test):
    xgb = XGBClassifier(colsample_bytree=0.5,
                        learning_rate=0.5,
                        max_depth=10,
                        min_child_weight=1,
                        n_estimators=1500,
                        scale_pos_weight=1,
                        objective='reg:squarederror',
                        subsample=0.6,
                        gamma=0.3,
                        booster='gbtree')

    xgb.fit(np.array(X), np.array(y))


def main():
    X, X_val, X_test, y, y_val, y_test = data_prep()
    train_models(X, X_val, X_test, y, y_val, y_test)


def data_prep():
    """
        Prepare data-set for modelling
        """
    train_data = pd.DataFrame(load_train_csv("../data"))
    test_data = pd.DataFrame(load_public_test_csv("../data"))
    student_metadata = load_student_meta_csv("../data")
    question_metadata = load_question_meta_csv("../data")

    # train_data = pd.merge(old_train_data, student_metadata, on="user_id")
    val_data = pd.DataFrame(load_valid_csv("../data"))
    # val_data = pd.merge(old_val_data, student_metadata, on="user_id")
    X = []
    for i in range(len(train_data['user_id'])):
        X.append([train_data['user_id'][i], train_data['question_id'][i]])

    y = train_data['is_correct']

    X_val = []
    for i in range(len(val_data['user_id'])):
        X_val.append([val_data['user_id'][i], val_data['question_id'][i]])
    y_val = val_data['is_correct']

    X_test = []
    for i in range(len(test_data['user_id'])):
        X_test.append([test_data['user_id'][i], test_data['question_id'][i]])
    y_test = test_data['is_correct']

    return X, X_val, X_test, y, y_val, y_test


if __name__ == "__main__":
    # data_prep()
    main()
