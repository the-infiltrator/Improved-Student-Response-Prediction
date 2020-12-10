import numpy as np
from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix
from sklearn.utils import resample
from utils import *
import item_response


class BaggedPredictor(ABC):

    def __init__(self, data, num_questions, num_students):
        self.data, self.weights = generate_bagged_sparse_dataset(data,
                                                                 num_questions,
                                                                 num_students)

    @abstractmethod
    def train(self, *args):
        pass

    @abstractmethod
    def predict(self, data, question_id, user_id):
        # return np.random.choice(2, data.shape[0])
        pass


class IRTPredictor(BaggedPredictor):

    def __init__(self, data, num_questions, num_students):
        super().__init__(data, num_questions, num_students)
        self.theta = np.zeros(num_students)
        self.beta = np.zeros(num_questions)

    def train(self, **kwargs):
        self.theta, self.beta = item_response.weighted_train(self.data,
                                                             self.theta,
                                                             self.beta,
                                                             self.weights,
                                                             **kwargs)

    def predict(self, data, question_id, user_id):
        x = (self.theta[user_id] - self.beta[question_id]).sum()
        p_a = item_response.sigmoid(x)
        return p_a
        # pred = []
        # for i, q in enumerate(data["question_id"]):
        #     u = data["user_id"][i]
        #     x = (self.theta[u] - self.beta[q]).sum()
        #     p_a = item_response.sigmoid(x)
        #     pred.append(p_a >= 0.5)


def generate_bagged_sparse_dataset(data, num_questions, num_students):
    n = len(data["question_id"])

    sparse = np.empty((num_students, num_questions))
    sparse.fill(np.nan)
    weights = np.zeros((num_students, num_questions))

    idx = np.random.choice(n, n)
    for id in idx:
        sparse[data["user_id"][id], data["question_id"][id]] = \
            data["is_correct"][id]
        weights[data["user_id"][id], data["question_id"][id]] += 1

    return csr_matrix(sparse), csr_matrix(weights)


def predict_bagged(data, question_id, user_id, predictors):
    predictions = np.array(
        [predictor.predict(data, question_id, user_id) for predictor in
         predictors])
    return np.mean(predictions) >= 0.5


def evaluate(data, predictors):
    pred = []
    for i, q in enumerate(data["question_id"]):
        pred.append(predict_bagged(data, q, data["user_id"][i], predictors))
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    # Get Data
    np.random.seed(20201200)
    train_data = load_train_csv("../data")
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    num_students, num_questions = sparse_matrix.toarray().shape
    # Initialize Predictors
    first_predictor = IRTPredictor(train_data, num_questions, num_students)
    second_predictor = IRTPredictor(train_data, num_questions, num_students)
    third_predictor = IRTPredictor(train_data, num_questions, num_students)

    # Train Predictors
    first_predictor.train(lr=0.005, iterations=250)
    second_predictor.train(lr=0.005, iterations=250)
    third_predictor.train(lr=0.005, iterations=250)

    # Evaluate Predictors
    val_acc = evaluate(val_data,
                           [first_predictor, second_predictor, third_predictor])

    test_acc = evaluate(test_data,
                           [first_predictor, second_predictor, third_predictor])
    print(
        f'\n###################################################################################\n'
        f'                                TRAINING COMPLETE                                  \n'
        f'                     Final Validation Accuracy = {val_acc}\n'
        f'                     Final Test Accuracy = {test_acc} \n'
        f'###################################################################################\n')


if __name__ == "__main__":
    main()
