from utils import *
from check_grad import check_grad

import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import pandas as pd


def sigmoid(x):
    """ Apply sigmoid function.
    """
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

    :param sparse_matrix: a sparse matrix of student responses
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # Implement the function as described in the docstring.             #
    #####################################################################
    C = np.nan_to_num(sparse_matrix.toarray())
    diff_mat = _difference_matrix(sparse_matrix, theta, beta)
    diff_mat = remove_nan_indices(sparse_matrix, diff_mat)
    cdiff = np.sum(np.multiply(C, diff_mat))

    log_diff = np.sum(
        remove_nan_indices(sparse_matrix, np.log(1 + np.exp(diff_mat))))
    log_lklihood = cdiff - log_diff
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(sparse_matrix, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """

    #####################################################################
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


def make_sparse(data, num_students, num_questions):
    """
    Converts the dictionary format of the data to a sparse matrix of
    observations.
    """
    n = len(data["question_id"])
    sparse = np.empty((num_students, num_questions))
    sparse.fill(np.nan)
    for id in range(n):
        sparse[data["user_id"][id], data["question_id"][id]] = \
            data["is_correct"][id]
    return csr_matrix(sparse)


def weighted_neg_log_liklihood(sparse_matrix, theta, beta, weights):
    """
    Compute the negative log-likelihood using the weights for each obsevation.

    :param sparse_matrix: a sparse matrix of student responses
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    C = np.nan_to_num(sparse_matrix.toarray())
    observation_weights = np.nan_to_num(weights.toarray())
    diff_mat = _difference_matrix(sparse_matrix, theta,
                                  beta) * observation_weights
    diff_mat = remove_nan_indices(sparse_matrix, diff_mat)
    cdiff = np.sum(np.multiply(C * observation_weights, diff_mat))

    log_diff = np.sum(
        remove_nan_indices(sparse_matrix, np.log(1 + np.exp(diff_mat))))
    log_lklihood = cdiff - log_diff
    return log_lklihood


def weighted_update_theta_beta(sparse_matrix, lr, theta, beta, weights):
    """ Update theta and beta using gradient descent, and considering the
    weights for each entry of the sparse matrix.

    :param sparse_matrix: a sparse matrix of student responses
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :param weights: a matrix of weights for each entry
    :return: tuple of vectors
    """
    C = np.nan_to_num(sparse_matrix.toarray())
    observation_weights = weights.toarray()

    for i in range(1):
        sig_diff_mat = sigmoid(_difference_matrix(sparse_matrix, theta,
                                                  beta))
        sig_diff_mat = remove_nan_indices(sparse_matrix, sig_diff_mat)
        dl_dtheta = (np.sum(C * observation_weights, axis=1) - np.sum(
            sig_diff_mat * observation_weights, axis=1))
        theta += lr * dl_dtheta

        sig_diff_mat2 = sigmoid(_difference_matrix(sparse_matrix, theta,
                                                   beta))
        sig_diff_mat2 = remove_nan_indices(sparse_matrix, sig_diff_mat2)
        dl_dbeta = (-np.sum(C * observation_weights, axis=0) + np.sum(
            sig_diff_mat2 * observation_weights, axis=0))
        beta += lr * dl_dbeta

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def weighted_train(sparse_matrix, theta, beta, weights, lr, iterations):
    """
    Train IRT using weights
    """
    for i in range(iterations):
        print(f"IRT: Iteration #{i + 1}")
        theta, beta = weighted_update_theta_beta(sparse_matrix, lr, theta, beta,
                                                 weights)
    return theta, beta


def train(sparse_matrix, theta, beta, weights, lr, iterations):
    """
    Train IRT using hyperparameters
    """
    # TODO : Check if used
    for i in range(iterations):
        print(f"IRT: Iteration #{i + 1}")
        theta, beta = update_theta_beta(sparse_matrix, lr, theta, beta)
    return theta, beta


def weighted_irt(sparse_matrix, val_data, lr, iterations, weights):
    """ Train IRT model using weights for each observation.

    :param sparse_matrix: a sparse matrix of student responses
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
        neg_lld = weighted_neg_log_liklihood(sparse_matrix, theta=theta,
                                             beta=beta, weights=weights)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = weighted_update_theta_beta(sparse_matrix, lr, theta, beta,
                                                 weights)

    return theta, beta, val_acc_lst


def irt(sparse_matrix, val_data, lr, iterations):
    """ Train IRT model.

    :param sparse_matrix: a sparse matrix of student responses
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    print(
        f"#########################################################################################\n"
        f"                         TRAINING ITEM RESPONSE THEORY MODEL                                \n"
        f"#########################################################################################\n")
    theta = np.zeros(sparse_matrix.shape[0])
    beta = np.zeros(sparse_matrix.shape[1])
    valid_matrix = make_sparse(val_data, *sparse_matrix.shape)
    val_loss_lst = []
    training_loss_lst = []
    val_acc = []
    for i in range(iterations):
        neg_lld = neg_log_likelihood(sparse_matrix, theta=theta, beta=beta)
        training_loss_lst.append(neg_lld)
        val_loss_lst.append(neg_log_likelihood(valid_matrix, theta=theta, beta=beta))
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc.append(score)
        print(f"Iteration: {i} NLLK: {neg_lld} \t Score: {score}")
        theta, beta = update_theta_beta(sparse_matrix, lr, theta, beta)

    return theta, beta, val_acc, val_loss_lst, training_loss_lst


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
    """
    Function that calculates log likelihood and dl_dtheta
    """
    ll = -neg_log_likelihood(sparse_matrix, theta, beta)
    C = np.nan_to_num(sparse_matrix.toarray())
    sig_diff_mat = sigmoid(_difference_matrix(sparse_matrix, theta, beta))
    sig_diff_mat = remove_nan_indices(sparse_matrix, sig_diff_mat)
    dl_dtheta = (np.sum(C, axis=1) - np.sum(sig_diff_mat, axis=1))
    return ll, dl_dtheta


def check_grad_beta(beta, sparse_matrix, theta):
    """
    Function that calculates log likelihood and dl_dbeta
    """
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


def gen_plots(theta, beta):
    """
    Generate the plots of p(c_{ij}=1|theta,beta) for trained theta and beta
    """
    betas = np.random.choice(beta, 5)

    theta = np.sort(theta)
    probabilities = []
    # [684, 559, 1653, 1216, 835]

    for i, beta_j in enumerate(betas):
        probabilities.append(sigmoid(theta - beta_j))

    plt.style.use('ggplot')
    plt.style.use('seaborn-paper')
    plt.xlabel('Theta')
    plt.ylabel('p(c_{ij}=1)')
    colors = ["red", "blue", "orange", "green", "purple"]

    for i in range(len(probabilities)):
        plt.plot(theta, probabilities[i], color=colors[i], label=f"Question {beta.values.tolist().index(betas[i])+1}")
    plt.legend(loc="upper left")
    plt.show()


def plot_costs(val_c, training_c):
    """
    Plot the accuracy on the validation data as a function of k.

    :param k_set: array of k values
    :param accuracy_set: array of validation accuracy for each k
    :return: None
    """
    plt.style.use('ggplot')
    plt.style.use('seaborn-paper')

    plt.xlabel('Iterations')
    plt.ylabel('Training Log-Likelihood')
    plt.plot(range(len(training_c)), training_c, color="darkcyan")
    plt.savefig("irt_train.png")
    plt.close()

    plt.xlabel('Iterations')
    plt.ylabel('Validation Log-Likelihood')
    plt.plot(range(len(val_c)), val_c, color="firebrick")
    plt.savefig("irt_validation.png")
    plt.close()


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")

    test_data = load_public_test_csv("../data")
    # #####################################################################
    # # Tune learning rate and number of iterations. With the implemented #
    # # code, report the validation and test accuracy.                    #
    # #####################################################################
    run_check_grad_theta(sparse_matrix[:50, :75])
    run_check_grad_beta(sparse_matrix[:50, :75])

    # theta, beta, val_acc, val_loss_lst, training_loss_lst = irt(sparse_matrix, val_data, 0.01, 280)
    # theta, beta, val_acc, val_loss_lst, training_loss_lst = irt(sparse_matrix,
    #                                                             val_data, 0.01,
    #                                                             25)
    # TODO: Remove
    theta = pd.read_csv("theta.csv")["Theta"]
    beta = pd.read_csv("beta.csv")["Beta"]
    # Make training plots
    # plot_costs(val_loss_lst, training_loss_lst)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    #####################################################################
    # Implement part (c)                                                #

    # print(
    #     f'\n###################################################################################\n'
    #     f'                                TRAINING COMPLETE                                  \n'
    #     f'                     Final Validation Accuracy = {val_acc[-1]}\n'
    #     f'                     Final Test Accuracy = {evaluate(test_data, theta, beta)} \n'
    #     f'###################################################################################\n')

    #####################################################################
    gen_plots(theta, beta)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
