from utils import *
from sklearn.impute import KNNImputer
import numpy as np

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))

def _difference_matrix(sparse_matrix,theta, beta):
    """
    Generate a difference matrix D, with D_{ij} = theta_i - beta_j
    """
    # C = np.nan_to_num(sparse_matrix.toarray())
    C = sparse_matrix
    theta_matrix = np.tile(theta, (C.shape[1], 1)).T
    beta_matrix = np.tile(beta, (C.shape[0], 1))
    diff = theta_matrix-beta_matrix
    return diff

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
    # C  = np.nan_to_num(sparse_matrix.toarray())
    C = sparse_matrix
    thetadotc = np.sum(theta @ C)
    cdotbeta = np.sum(C@beta)
    cdiff = thetadotc-cdotbeta
    diff_mat = _difference_matrix(sparse_matrix,theta,beta)
    log_diff = np.sum(np.log(1+np.exp(diff_mat)))
    log_lklihood = cdiff - log_diff

    # log_lklihood = np.nansum(np.array([sparse_matrix[i, j] * (theta[i] - beta[j]) - np.log(1 + np.exp(theta[i] - beta[j])) for i in
    # range(sparse_matrix.shape[0]) for j in range(sparse_matrix.shape[1])]))
    # tf.reduce_sum(C * tf.log(tf.sigmoid(theta - beta)) + (1 - C) * tf.log(1 - tf.sigmoid(theta - a)))
    # print(log_lklihood)
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
    # C = np.nan_to_num(sparse_matrix.toarray())
    C = sparse_matrix

    for i in range(100):
        sig_diff_mat = sigmoid(_difference_matrix(sparse_matrix, theta, beta))
        # print(np.nansum(C, axis=1).shape)
        dl_dtheta = -(np.nansum(C, axis=1) - np.nansum(sig_diff_mat, axis=1))
        dl_dbeta = -(-np.nansum(C, axis=0) + np.nansum(sig_diff_mat, axis=0))
        theta = theta - lr * dl_dtheta
        beta = beta - lr * dl_dbeta

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
    # TODO: Initialize theta and beta.
    theta = np.zeros(sparse_matrix.shape[0])
    beta = np.zeros(sparse_matrix.shape[1])

    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(sparse_matrix, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(sparse_matrix, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
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


def main():

    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    # print(sparse_matrix[1, 2])
    test_data = load_public_test_csv("../data")

    nbrs = KNNImputer(n_neighbors=11)
    # We use NaN-Euclidean distance measure.
    C = sparse_matrix.toarray()
    sparse_matrix = nbrs.fit_transform(C)
    # #####################################################################
    # # TODO:                                                             #
    # # Tune learning rate and number of iterations. With the implemented #
    # # code, report the validation and test accuracy.                    #
    # #####################################################################
    irt(sparse_matrix, test_data, 0.01, 20)
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
