import numpy as np
from abc import ABC, abstractmethod

import torch
from torch.autograd import Variable
from scipy.sparse import csr_matrix
from utils import *
import item_response
import neural_network
import two_item_response


class BaggedPredictor(ABC):

    def __init__(self, data, num_questions, num_students):
        self.data, self.weights = generate_bagged_sparse_dataset(data,
                                                                 num_questions,
                                                                 num_students)

    @abstractmethod
    def train(self, *args):
        pass

    @abstractmethod
    def predict(self, data, question_id, user_id, sparse):
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

    def predict(self, data, question_id, user_id, sparse):
        x = (self.theta[user_id] - self.beta[question_id]).sum()
        p_a = item_response.sigmoid(x)
        return p_a


class AEPredictor(BaggedPredictor):
    def __init__(self, data, num_questions, num_students, k):
        super().__init__(data, num_questions, num_students)

        self.model = neural_network.AutoEncoder(num_questions, k=k)

        self.data = self.data.toarray()
        self.zero_train_data = self.data.copy()
        self.zero_train_data[np.isnan(self.data)] = 0

        self.train_data = torch.FloatTensor(self.data)
        self.zero_train_data = torch.FloatTensor(self.zero_train_data)
        self.weights = torch.FloatTensor(self.weights.toarray())

    def train(self, **kwargs):
        neural_network.train_ensemble(self.model,
                                      train_data=self.train_data,
                                      zero_train_data=self.zero_train_data,
                                      weights=self.weights, **kwargs)

    def predict(self, data, question_id, user_id, sparse):
        inputs = Variable(sparse[user_id]).unsqueeze(0)
        output = self.model(inputs)

        guess = output[0][question_id].item()
        return guess


class TwoParamIRTPredictor(BaggedPredictor):
    def __init__(self, data, num_questions, num_students):
        super().__init__(data, num_questions, num_students)
        self.theta = np.zeros(num_students)
        self.beta = np.zeros(num_questions)
        self.alpha = np.ones(num_questions)

    def train(self, **kwargs):
        self.theta, self.beta = two_item_response.weighted_train(self.data,
                                                                 self.theta,
                                                                 self.beta,
                                                                 self.alpha,
                                                                 self.weights,
                                                                 **kwargs)

    def predict(self, data, question_id, user_id, sparse):
        x = (self.alpha[question_id] * (self.theta[user_id] - self.beta[question_id])).sum()
        p_a = item_response.sigmoid(x)
        return p_a


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


def predict_bagged(data, question_id, user_id, predictors, sparse):
    predictions = np.array(
        [predictor.predict(data, question_id, user_id, sparse) for predictor in
         predictors])
    return np.mean(predictions) >= 0.5


def evaluate(data, predictors, sparse):
    pred = []
    for i, q in enumerate(data["question_id"]):
        pred.append(
            predict_bagged(data, q, data["user_id"][i], predictors, sparse))
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

    valid_sparse, _ = neural_network.make_sparse(val_data, num_students,
                                                 num_questions)
    test_sparse, _ = neural_network.make_sparse(test_data, num_students,
                                                num_questions)

    # Initialize Predictors
    first_predictor = IRTPredictor(train_data, num_questions, num_students)
    second_predictor = IRTPredictor(train_data, num_questions, num_students)
    third_predictor = IRTPredictor(train_data, num_questions, num_students)
    # first_predictor = AEPredictor(train_data, num_questions, num_students, k=10)
    # second_predictor = AEPredictor(train_data, num_questions, num_students,
    #                                k=10)
    # third_predictor = AEPredictor(train_data, num_questions, num_students, k=10)
    # Train Predictors)
    # first_predictor.train(lr=0.021, lamb=0.01, num_epoch=10)
    # second_predictor.train(lr=0.021, lamb=0.01, num_epoch=10)
    # third_predictor.train(lr=0.021, lamb=0.01, num_epoch=10)
    # print(neural_network.evaluate(first_predictor.model, first_predictor.zero_train_data, val_data))
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
