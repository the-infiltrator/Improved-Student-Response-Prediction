from utils import *
from check_grad import check_grad

import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    # return np.exp(x) / (1 + np.exp(x))
    # return 1 / (1 + np.exp(-x))
    return np.exp(-np.logaddexp(0, -x))


def _difference_matrix(sparse_matrix, theta, beta):
    """
    Generate a difference matrix D, with D_{ij} = theta_i - beta_j
    """
    C = np.nan_to_num(sparse_matrix.toarray())
    theta_matrix = np.tile(theta, (C.shape[1], 1)).T
    beta_matrix = np.tile(beta, (C.shape[0], 1))
    diff = theta_matrix - beta_matrix
    # diff[np.isnan(sparse_matrix.toarray())] = 0
    return diff


def remove_nan_indices(sparse_matrix, matrix):
    """
    PRECONDITION: sparse_matrix.shape == matrix.shape
    For every [i,j] in sparse_matrix, if the value is NaN/Missing,
    make matrix[i,j] = 0

    Does not modify the original matrix.
    """
    m = np.copy(matrix)
    m[np.isnan(sparse_matrix.toarray())] = 0
    return m


def neg_log_likelihood(sparse_matrix, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    C = np.nan_to_num(sparse_matrix.toarray())
    diff_mat = _difference_matrix(sparse_matrix, theta, beta)
    diff_mat = remove_nan_indices(sparse_matrix, diff_mat)
    cdiff = np.sum(np.multiply(C, diff_mat))

    log_diff = np.sum(np.log(1 + np.exp(diff_mat)))
    log_lklihood = cdiff - log_diff
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(sparse_matrix, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    C = np.nan_to_num(sparse_matrix.toarray())

    for i in range(1):
        sig_diff_mat = sigmoid(_difference_matrix(sparse_matrix, theta, beta))
        sig_diff_mat = remove_nan_indices(sparse_matrix, sig_diff_mat)
        dl_dtheta = (np.sum(C, axis=1) - np.sum(sig_diff_mat, axis=1))
        theta += lr * dl_dtheta

        sig_diff_mat2 = sigmoid(_difference_matrix(sparse_matrix, theta, beta))
        sig_diff_mat2 = remove_nan_indices(sparse_matrix, sig_diff_mat2)
        dl_dbeta = (-np.sum(C, axis=0) + np.sum(sig_diff_mat2, axis=0))
        beta += lr * dl_dbeta

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(sparse_matrix, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    theta = np.zeros(sparse_matrix.shape[0])
    beta = np.zeros(sparse_matrix.shape[1])

    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(sparse_matrix, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(sparse_matrix, lr, theta, beta)

    return theta, beta, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def check_grad_theta(theta, sparse_matrix, beta):
    ll = -neg_log_likelihood(sparse_matrix, theta, beta)
    C = np.nan_to_num(sparse_matrix.toarray())
    sig_diff_mat = sigmoid(_difference_matrix(sparse_matrix, theta, beta))
    sig_diff_mat = remove_nan_indices(sparse_matrix, sig_diff_mat)
    dl_dtheta = (np.sum(C, axis=1) - np.sum(sig_diff_mat, axis=1))
    return ll, dl_dtheta


def check_grad_beta(beta, sparse_matrix, theta):
    ll = -neg_log_likelihood(sparse_matrix, theta, beta)
    C = np.nan_to_num(sparse_matrix.toarray())
    sig_diff_mat = sigmoid(_difference_matrix(sparse_matrix, theta, beta))
    sig_diff_mat = remove_nan_indices(sparse_matrix, sig_diff_mat)
    dl_dbeta = (-np.sum(C, axis=0) + np.sum(sig_diff_mat, axis=0))
    return ll, dl_dbeta


def run_check_grad_theta(sparse_matrix):
    theta = np.zeros(sparse_matrix.shape[0])
    beta = np.zeros(sparse_matrix.shape[1])
    diff = check_grad(check_grad_theta,
                      theta,
                      0.001,
                      sparse_matrix,
                      beta)
    print("diff theta=", diff)


def run_check_grad_beta(sparse_matrix):
    theta = np.zeros(sparse_matrix.shape[0])
    beta = np.zeros(sparse_matrix.shape[1])
    diff = check_grad(check_grad_beta,
                      beta,
                      0.001,
                      sparse_matrix,
                      theta)
    print("diff beta=", diff)


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    # #####################################################################
    # # TODO:                                                             #
    # # Tune learning rate and number of iterations. With the implemented #
    # # code, report the validation and test accuracy.                    #
    # #####################################################################
    run_check_grad_theta(sparse_matrix[:50, :75])
    run_check_grad_beta(sparse_matrix[:50, :75])
    irt(sparse_matrix, val_data, 0.001, 100)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    #####################################################################
    # TODO:                                                             #
    # Implement part (c)                                                #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
