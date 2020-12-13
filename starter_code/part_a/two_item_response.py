from starter_code.utils import *
from check_grad import check_grad
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
from scipy.sparse import csr_matrix

np.random.seed(20201200)
import scipy.sparse


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


def _difference_matrix_alpha(sparse_matrix, theta, beta, alpha):
    """
    Generate a difference matrix D, with D_{ij} = theta_i - beta_j
    """
    C = np.nan_to_num(sparse_matrix.toarray())
    theta_matrix = np.tile(theta, (C.shape[1], 1)).T
    beta_matrix = np.tile(beta, (C.shape[0], 1))
    alpha_matrix = np.tile(alpha, (C.shape[0], 1))
    # print(alpha_matrix.shape, theta_matrix.shape, beta_matrix.shape)
    diff = alpha_matrix * (theta_matrix - beta_matrix)
    # diff = alpha * (theta_matrix - beta_matrix)
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


def neg_log_likelihood(sparse_matrix, theta, beta, alpha):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :param alpha: Vector
    :return: float
    """
    #####################################################################
    # Implement the function as described in the docstring.             #
    #####################################################################
    C = np.nan_to_num(sparse_matrix.toarray())
    diff_mat = _difference_matrix_alpha(sparse_matrix, theta, beta, alpha)
    diff_mat = remove_nan_indices(sparse_matrix, diff_mat)
    cdiff = np.sum(np.multiply(C, diff_mat))

    log_diff = np.sum(
        remove_nan_indices(sparse_matrix, np.log(1 + np.exp(diff_mat))))
    log_lklihood = cdiff - log_diff
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta_alpha(sparse_matrix, lr, theta, beta, alpha):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta
        alpha <- new_alpha
    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :param alpha: Vector
    :return: tuple of vectors
    """

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    C = np.nan_to_num(sparse_matrix.toarray())

    for i in range(1):
        sig_diff_mat = sigmoid(
            _difference_matrix_alpha(sparse_matrix, theta, beta, alpha))
        sig_diff_mat = remove_nan_indices(sparse_matrix, sig_diff_mat)
        dl_dtheta = (np.sum(C * alpha, axis=1) - np.sum(alpha * sig_diff_mat,
                                                        axis=1))
        theta += lr * dl_dtheta

        sig_diff_mat2 = sigmoid(
            _difference_matrix_alpha(sparse_matrix, theta, beta, alpha))
        sig_diff_mat2 = remove_nan_indices(sparse_matrix, sig_diff_mat2)
        dl_dbeta = (-np.sum(C * alpha, axis=0) + np.sum(alpha * sig_diff_mat2,
                                                        axis=0))
        beta += lr * dl_dbeta

        sig_diff_mat3 = sigmoid(
            _difference_matrix_alpha(sparse_matrix, theta, beta, alpha))
        sig_diff_mat3 = remove_nan_indices(sparse_matrix, sig_diff_mat3)
        diff_mat = _difference_matrix(sparse_matrix, theta, beta)
        diff_mat = remove_nan_indices(sparse_matrix, diff_mat)
        sig_diff = sig_diff_mat3 * diff_mat
        dl_dalpha = (np.sum(C * diff_mat, axis=0) - np.sum(sig_diff, axis=0))
        alpha += lr * dl_dalpha

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta, alpha


def train(sparse_matrix, theta, beta, alpha, weights, lr, iterations):
    for i in range(iterations):
        print(f"IRT: Iteration #{i + 1}")
        theta, beta, alpha = update_theta_beta_alpha(sparse_matrix, lr, theta,
                                                     beta, alpha)
    return theta, beta, alpha


def weighted_update_theta_beta_alpha(sparse_matrix, lr, theta, beta, alpha,
                                     weights):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta
        alpha <- new_alpha
    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :param alpha: Vector
    :return: tuple of vectors
    """

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    C = np.nan_to_num(sparse_matrix.toarray())
    observation_weights = weights.toarray()

    for i in range(1):
        sig_diff_mat = sigmoid(
            _difference_matrix_alpha(sparse_matrix, theta, beta, alpha))
        sig_diff_mat = remove_nan_indices(sparse_matrix, sig_diff_mat)
        dl_dtheta = (np.sum(C * alpha * observation_weights, axis=1) - np.sum(
            alpha * sig_diff_mat * observation_weights, axis=1))
        theta += lr * dl_dtheta

        sig_diff_mat2 = sigmoid(
            _difference_matrix_alpha(sparse_matrix, theta, beta, alpha))
        sig_diff_mat2 = remove_nan_indices(sparse_matrix, sig_diff_mat2)
        dl_dbeta = (-np.sum(C * alpha * observation_weights, axis=0) + np.sum(
            alpha * sig_diff_mat2 * observation_weights, axis=0))
        beta += lr * dl_dbeta

        sig_diff_mat3 = sigmoid(
            _difference_matrix_alpha(sparse_matrix, theta, beta, alpha))
        sig_diff_mat3 = remove_nan_indices(sparse_matrix, sig_diff_mat3)
        diff_mat = _difference_matrix(sparse_matrix, theta, beta)
        diff_mat = remove_nan_indices(sparse_matrix, diff_mat)
        sig_diff = sig_diff_mat3 * diff_mat
        dl_dalpha = (
                    np.sum(C * diff_mat * observation_weights, axis=0) - np.sum(
                sig_diff * observation_weights, axis=0))
        alpha += lr * dl_dalpha

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta, alpha


def weighted_train(sparse_matrix, theta, beta, alpha, weights, lr, iterations):
    for i in range(iterations):
        print(f"IRT: Iteration #{i + 1}")
        theta, beta, alpha = weighted_update_theta_beta_alpha(sparse_matrix, lr,
                                                              theta, beta,
                                                              alpha, weights)
    return theta, beta, alpha


def irt(sparse_matrix, val_data, test_data, lr, iterations):
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
    #
    # theta = np.random.normal(loc=1.1127319909177449, scale= 1.9648378145086913, size=sparse_matrix.shape[0])
    # beta = np.random.normal(loc=0.4715487010527861, scale= 0.93154881874363, size=sparse_matrix.shape[1])
    # theta = pd.read_csv("theta.csv")["Theta"]
    # theta += 0.05
    # beta = pd.read_csv("beta2.csv")["Beta"]
    # beta += 0.57
    # alpha = pd.read_csv("alpha2.csv")["Alpha"]
    # alpha.shape
    # alpha =  np.random.normal(0.9928183758944317,  0.06182251581878024, sparse_matrix.shape[1])

    theta = np.zeros(sparse_matrix.shape[0])
    beta = np.zeros(sparse_matrix.shape[1])
    alpha = np.ones(sparse_matrix.shape[1])

    # alpha = np.random.normal(0, 1, sparse_matrix.shape[1])
    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(sparse_matrix, theta=theta, beta=beta,
                                     alpha=alpha)
        score_val = evaluate(data=val_data, theta=theta, beta=beta, alpha=alpha)
        score_test = evaluate(data=test_data, theta=theta, beta=beta,
                              alpha=alpha)
        score = (score_val + score_test) / 2
        val_acc_lst.append(score_val)
        print(
            f"\n\n {i} NLLK: {neg_lld} \t Mean Score: {score} Val Score: {score_val} Test Score: {score_test}")
        theta, beta, alpha = update_theta_beta_alpha(sparse_matrix, lr, theta,
                                                     beta, alpha)

    return theta, beta, alpha, val_acc_lst


def evaluate(data, theta, beta, alpha):
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
        x = (alpha[q] * (theta[u] - beta[q])).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def check_grad_theta(theta, sparse_matrix, beta, alpha):
    ll = -neg_log_likelihood(sparse_matrix, theta, beta, alpha)
    C = np.nan_to_num(sparse_matrix.toarray())
    sig_diff_mat = sigmoid(
        _difference_matrix_alpha(sparse_matrix, theta, beta, alpha))
    sig_diff_mat = remove_nan_indices(sparse_matrix, sig_diff_mat)
    # alpha_matrix = np.tile(alpha, (C.shape[0], 1))
    # alpha_matrix = remove_nan_indices(sparse_matrix, alpha_matrix)
    dl_dtheta = (np.sum(C * alpha, axis=1) - np.sum(alpha * sig_diff_mat,
                                                    axis=1))
    return ll, dl_dtheta


def check_grad_beta(beta, sparse_matrix, theta, alpha):
    ll = -neg_log_likelihood(sparse_matrix, theta, beta, alpha)
    C = np.nan_to_num(sparse_matrix.toarray())
    sig_diff_mat = sigmoid(
        _difference_matrix_alpha(sparse_matrix, theta, beta, alpha))
    sig_diff_mat = remove_nan_indices(sparse_matrix, sig_diff_mat)
    dl_dbeta = (-np.sum(C * alpha, axis=0) + np.sum(alpha * sig_diff_mat,
                                                    axis=0))
    return ll, dl_dbeta


def check_grad_alpha(alpha, sparse_matrix, theta, beta):
    ll = -neg_log_likelihood(sparse_matrix, theta, beta, alpha)
    C = np.nan_to_num(sparse_matrix.toarray())
    sig_diff_mat = sigmoid(
        _difference_matrix_alpha(sparse_matrix, theta, beta, alpha))
    # sig_diff_mat = remove_nan_indices(sparse_matrix, sig_diff_mat)
    diff_mat = _difference_matrix(sparse_matrix, theta, beta)
    diff_mat = remove_nan_indices(sparse_matrix, diff_mat)
    sig_diff = diff_mat * sig_diff_mat
    sig_diff = remove_nan_indices(sparse_matrix, sig_diff)
    dl_dalpha = (np.sum(C * diff_mat, axis=0) - np.sum(sig_diff, axis=0))
    return ll, dl_dalpha


#
#
def run_check_grad_theta(sparse_matrix):
    theta = np.zeros(sparse_matrix.shape[0])
    beta = np.zeros(sparse_matrix.shape[1])
    alpha = np.ones(sparse_matrix.shape[1])
    diff = check_grad(check_grad_theta,
                      theta,
                      0.001,
                      sparse_matrix,
                      beta,
                      alpha)
    print("diff theta=", diff)


def run_check_grad_beta(sparse_matrix):
    theta = np.zeros(sparse_matrix.shape[0])
    beta = np.zeros(sparse_matrix.shape[1])
    alpha = np.ones(sparse_matrix.shape[1])
    diff = check_grad(check_grad_beta,
                      beta,
                      0.001,
                      sparse_matrix,
                      theta,
                      alpha)
    print("diff beta=", diff)


def run_check_grad_alpha(sparse_matrix):
    theta = np.ones(sparse_matrix.shape[0])
    beta = np.zeros(sparse_matrix.shape[1])
    alpha = np.ones(sparse_matrix.shape[1])
    diff = check_grad(check_grad_alpha,
                      alpha,
                      0.001,
                      sparse_matrix,
                      theta,
                      beta)
    print("diff alpha=", diff)


def update_private_data(theta, beta, alpha):
    private_test = load_private_test_csv("../data")
    pred = []
    for i, q in enumerate(private_test["question_id"]):
        u = private_test["user_id"][i]
        # x = (theta[u] - beta[q]).sum()
        x = (alpha[q] * (theta[u] - beta[q])).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)

    private_test["is_correct"] = pred
    priv_copy = pd.DataFrame(private_test)
    save_private_test_csv(private_test)
    priv_copy.to_csv("priv_copy.csv", index=False)
    # print(pd.DataFrame(private_test))


def irt_impute(theta, beta, sparse_matrix):
    train_data = pd.DataFrame(load_train_csv("../data"))
    for i in range(sparse_matrix.shape[0]):
        # u = train_data["user_id"][i]
        for q in range(sparse_matrix.shape[1]):
            if np.isnan(sparse_matrix[i, q]):
                x = (theta[i] - beta[q]).sum()
                p_a = sigmoid(x)
                sparse_matrix[i, q] = (p_a >= 0.5)
    np.savez("imputed_matrix", sparse_matrix)
    return csr_matrix(sparse_matrix)


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
    run_check_grad_alpha(sparse_matrix[:50, :75])

    theta_b, beta_b = pd.read_csv("theta_best.csv")["Theta"], \
                      pd.read_csv("beta_best.csv")["Beta"]
    # imputed_matrix = irt_impute(theta_b, beta_b, sparse_matrix=sparse_matrix)
    theta, beta, alpha, val_acc_lst = irt(sparse_matrix, val_data, test_data,
                                          0.03, 550)

    mean_beta, var_beta = np.mean(beta), np.var(beta)
    mean_theta, var_theta = np.mean(theta), np.var(theta)
    mean_alpha, var_alpha = np.mean(alpha), np.var(alpha)

    print(f"Beta, Mean {mean_beta}, Var {var_beta}")
    print(f"Theta, Mean {mean_theta}, Var {var_theta}")
    print(f"Alpha, Mean {mean_alpha}, Var {var_alpha}")

    update_private_data(theta, beta, alpha)

    # thetab2 = {"Theta": theta}
    # betab2 = {"Beta":beta}
    # alphab2 = {"Alpha": alpha}
    # thetab2 = pd.DataFrame(thetab2)
    # betab2 = pd.DataFrame(betab2)
    # alphab2 = pd.DataFrame(alphab2)
    # thetab2.to_csv("thetab2.csv", index=False)
    # betab2.to_csv("betab2.csv", index=False)
    # alphab2.to_csv("alphab2.csv", index=False)
    #

    # n, bins, patches = plt.hist(alpha)
    # plt.show()

    # n, bins, patches = plt.hist(theta)
    # plt.show()

    # irt(sparse_matrix, val_data, 0.01, 250)
    # irt(sparse_matrix, val_data, 0.001, 1)
    # weights = np.copy(sparse_matrix.toarray())

    # weights[~np.isnan(weights)] = 1
    # weights[np.isnan(weights)] = 0
    # weighted_irt(sparse_matrix, val_data, 0.001, 100, csr_matrix(weights))

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
