from scipy.stats import stats
from statsmodels.compat import scipy

from utils import *
from check_grad import check_grad
from scipy.stats import norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from girth import twopl_mml
from sklearn.linear_model import LogisticRegressionCV
import seaborn as sns

np.random.seed(20201200)



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


def neg_log_likelihood_two(sparse_matrix, theta, beta, alpha):
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


def one_param_irt(sparse_matrix, val_data, test_data, lr, iterations, zero_init=False):
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

    # 08/12 First Leaderboard Priors#####
    if zero_init:
        theta = np.zeros(sparse_matrix.shape[0])
        beta = np.zeros(sparse_matrix.shape[1])
        prior = "ZEROS"
    else:
        theta = np.random.normal(loc=0.8753958065096175, scale=1.7639304844742527, size=sparse_matrix.shape[0])

        beta = np.random.normal(loc=0.5232902213275956, scale=0.9104078214297008, size=sparse_matrix.shape[1])
        prior = "NORMAL"

    print(
        f"#########################################################################################\n"
        f"                   TRAINING 1-ITEM RESPONSE THEORY MODEL INIT: {prior}               \n"
        f"#########################################################################################\n")

    val_acc_lst = []
    neg_ll = []
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    for i in range(iterations):
        neg_lld = neg_log_likelihood(sparse_matrix, theta=theta, beta=beta)
        neg_ll.append(neg_lld)
        # val_acc_lst.append(score_val)
        # print(i)
        theta, beta = update_theta_beta(sparse_matrix, lr, theta, beta)
        score_val = evaluate(data=val_data, theta=theta, beta=beta)
        score_test = evaluate(data=test_data, theta=theta, beta=beta)
        val_acc_lst.append(score_val)
        # score_train = evaluate(data=train_data, theta=theta, beta=beta)
        # n, bins, patches = plt.hist(beta)
        mean_beta, var_beta = np.mean(beta), np.var(beta)
        mean_theta, var_theta = np.mean(theta), np.var(theta)

        # x = np.linspace(mean_beta - 3 * var_beta, mean_beta + 3 * var_beta, 100)
        # y = np.linspace(mean_theta - 3 * var_theta, mean_theta + 3 * var_theta, 100)
        # # f1 = f = plt.figure(1)

        # plt.plot(x, norm.pdf(x,  mean_beta , var_beta))

        # plt.plot(y, norm.pdf(y, mean_theta, var_theta))

        score = (score_val + score_test) / 2
        print(
            f"{i} NLLK: {neg_lld} \t Mean Score: {score} Val Score: {score_val} Test Score: {score_test}")
    # update_private_data(theta, beta)
    # n, bins, patches = plt.hist(theta)
    # plt.suptitle("Convergence of Beta")
    # plt.suptitle("Convergence of Theta")
    # plt.show()

    # print(f"Beta, Mean {mean_beta}, Var {var_beta}")
    # print(f"Theta, Mean {mean_theta}, Var {var_theta}")

    # sns.displot(beta, hist=True, kde=True,
    #              bins=int(180 / 5), color='darkblue',
    #              hist_kws={'edgecolor': 'black'})
    #
    # sns.displot(theta, hist=True, kde=True,
    #              bins=int(180 / 5), color='darkblue',
    #              hist_kws={'edgecolor': 'black'})

    update_private_data(theta, beta)
    return theta, beta, val_acc_lst, neg_ll


def one_param_irt_slower(theta, beta, sparse_matrix, val_data, test_data, lr, iterations):
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
    val_acc_lst = []
    neg_ll = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(sparse_matrix, theta=theta, beta=beta)
        neg_ll.append(neg_lld)
        score_val = evaluate(data=val_data, theta=theta, beta=beta)
        score_test = evaluate(data=test_data, theta=theta, beta=beta)
        # score_train = evaluate(data=train_data, theta=theta, beta=beta)
        score = (score_val + score_test) / 2
        val_acc_lst.append(score_val)
        print(f"{i} NLLK: {neg_lld} \t Mean Score: {score} Val Score: {score_val} Test Score: {score_test}")
        theta, beta = update_theta_beta(sparse_matrix, lr, theta, beta)
    # update_private_data(theta, beta)
    # n, bins, patches = plt.hist(theta)
    # plt.show()
    mean_beta, var_beta = np.mean(beta), np.var(beta)
    mean_theta, var_theta = np.mean(theta), np.var(theta)

    print(f"\n Beta, Mean {mean_beta}, Var {var_beta}")
    print(f"Theta, Mean {mean_theta}, Var {var_theta}")
    # n, bins, patches = plt.hist(beta)
    # plt.show()
    update_private_data(theta, beta)
    return theta, beta, val_acc_lst, neg_ll


def two_param_irt(theta, beta, sparse_matrix, val_data, test_data, lr, iterations, zero_init=False):
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
    # alpha = pd.read_csv("alpha2.csv")["Alpha"]
    # alpha.shape
    # alpha = np.ones(sparse_matrix.shape[1])
    # theta = np.zeros(sparse_matrix.shape[0])
    # beta = np.zeros(sparse_matrix.shape[1])
    # alpha = np.ones(sparse_matrix.shape[1])
    if zero_init:
        theta = np.zeros(sparse_matrix.shape[0])
        beta = np.zeros(sparse_matrix.shape[1])
        alpha = np.ones(sparse_matrix.shape[1])
        prior = "ZEROS"
    else:
        alpha = np.random.normal(1, 0.005, sparse_matrix.shape[1])
        prior = "NORMAL"

    print(
        f"#########################################################################################\n"
        f"                   TRAINING 2-ITEM RESPONSE THEORY MODEL INIT: {prior}               \n"
        f"#########################################################################################\n")
    val_acc_lst = []
    neg_ll = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood_two(sparse_matrix, theta=theta, beta=beta,
                                     alpha=alpha)
        neg_ll.append(neg_lld)
        score_val = evaluate_two_irt(data=val_data, theta=theta, beta=beta, alpha=alpha)
        score_test = evaluate_two_irt(data=test_data, theta=theta, beta=beta,
                              alpha=alpha)
        score = (score_val + score_test) / 2
        val_acc_lst.append(score_val)
        print(
            f"{i} NLLK: {neg_lld} \t Mean Score: {score} Val Score: {score_val} Test Score: {score_test}")
        theta, beta, alpha = update_theta_beta_alpha(sparse_matrix, lr, theta,
                                                     beta, alpha)

    return theta, beta, alpha, val_acc_lst, neg_ll


def update_private_data(theta, beta):
    private_test = load_private_test_csv("../data")
    pred = []
    for i, q in enumerate(private_test["question_id"]):
        u = private_test["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)

    private_test["is_correct"] = pred
    priv_copy = pd.DataFrame(private_test)
    save_private_test_csv(private_test)
    priv_copy.to_csv("priv_copy.csv", index=False)
    # print(pd.DataFrame(private_test))


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


def evaluate_two_irt(data, theta, beta, alpha):
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


def irt_plus(sparse_matrix, val_data, test_data):
    print(
        f"#########################################################################################\n"
        f"                         TRAINING ITEM RESPONSE THEORY++ MODEL                                \n"
        f"#########################################################################################\n")
    validation_accuracies = []
    negative_log_likelihoods = []
    theta, beta, val_acc, nll = one_param_irt(sparse_matrix, val_data, test_data, 0.01, 278)
    validation_accuracies.extend(val_acc)
    negative_log_likelihoods.extend(nll)
    theta, beta, val_acc, nll = one_param_irt_slower(theta, beta, sparse_matrix, val_data, test_data, 0.001, 31)
    validation_accuracies.extend(val_acc)
    negative_log_likelihoods.extend(nll)
    _, _, _, val_acc, nll = two_param_irt(theta, beta, sparse_matrix, val_data, test_data, 0.001, 143)
    validation_accuracies.extend(val_acc)
    negative_log_likelihoods.extend(nll)
    return validation_accuracies, negative_log_likelihoods


def plot_comparison(losses, accuracies):
    plt.style.use('ggplot')
    plt.style.use('seaborn-paper')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Training Loss')
    labels = ["1-Param Gaussian IRT",
              "1-Param Zero IRT",
              "2-Param Gaussian IRT",
              "2-Param Zero IRT",
              "IRT++"]
    colors = ["red", "blue", "orange", "green", "purple"]
    for i in range(len(losses)):
        plt.plot(range(20, len(losses[i])), losses[i][20:], color=colors[i])
        plt.plot(range(20, len(losses[i])), losses[i][20:], color=colors[i], label=labels[i])

    plt.legend(loc="upper right")
    plt.show()
    plt.close()

    plt.style.use('ggplot')
    plt.style.use('seaborn-paper')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Training Loss')

    for i in range(len(losses)):
        plt.plot(range(0, len(losses[i])), losses[i], color=colors[i])
        plt.plot(range(0, len(losses[i])), losses[i], color=colors[i], label=labels[i])

    plt.legend(loc="upper right")
    plt.show()
    plt.close()

    plt.style.use('ggplot')
    plt.style.use('seaborn-paper')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Validation Accuracy')

    for i in range(len(accuracies)):
        plt.plot(range(20, len(accuracies[i])), accuracies[i][20:], color=colors[i])
        plt.plot(range(20, len(accuracies[i])), accuracies[i][20:], color=colors[i], label=labels[i])

    plt.legend(loc="lower right")
    plt.show()
    plt.close()

    plt.style.use('ggplot')
    plt.style.use('seaborn-paper')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Validation Accuracy')

    for i in range(len(accuracies)):
        plt.plot(range(0, len(accuracies[i])), accuracies[i], color=colors[i])
        plt.plot(range(0, len(accuracies[i])), accuracies[i], color=colors[i], label=labels[i])
    plt.legend(loc="lower right")
    plt.show()
    plt.close()


def run_comparison(sparse_matrix, val_data, test_data, num_iterations):
    losses = []
    accuracies = []
    _, _, val_acc, neg_ll = one_param_irt(sparse_matrix, val_data, test_data, 0.01, num_iterations, zero_init=False)
    losses.append(neg_ll)
    accuracies.append(val_acc)

    _, _, val_acc, neg_ll = one_param_irt(sparse_matrix, val_data, test_data, 0.01, num_iterations, zero_init=True)
    losses.append(neg_ll)
    accuracies.append(val_acc)

    theta = np.random.normal(loc=0.8753958065096175, scale=1.7639304844742527, size=sparse_matrix.shape[0])
    beta = np.random.normal(loc=0.5232902213275956, scale=0.9104078214297008, size=sparse_matrix.shape[1])
    _, _, _, val_acc, neg_ll = two_param_irt(theta, beta, sparse_matrix, val_data, test_data, 0.01, num_iterations, zero_init=False)
    losses.append(neg_ll)
    accuracies.append(val_acc)

    _, _, _, val_acc, neg_ll = two_param_irt(theta, beta, sparse_matrix, val_data, test_data, 0.01, num_iterations, zero_init=True)
    losses.append(neg_ll)
    accuracies.append(val_acc)

    val_acc, neg_ll = irt_plus(sparse_matrix, val_data, test_data)
    losses.append(neg_ll)
    accuracies.append(val_acc)

    plot_comparison(losses, accuracies)


def main():
    train_data = pd.DataFrame(load_train_csv("../data"))
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = pd.DataFrame(load_valid_csv("../data"))
    test_data = pd.DataFrame(load_public_test_csv("../data"))

    # nbrs = KNNImputer(n_neighbors=5)
    # # We use NaN-Euclidean distance measure.
    # mat = nbrs.fit_transform(sparse_matrix.toarray())

    # #####################################################################
    # # Tune learning rate and number of iterations. With the implemented #
    # # code, report the validation and test accuracy.                    #
    # #####################################################################
    run_check_grad_theta(sparse_matrix[:50, :75])
    run_check_grad_beta(sparse_matrix[:50, :75])

    run_comparison(sparse_matrix, val_data, test_data, 452)
    # theta = pd.read_csv("theta.csv")["Theta"]
    # beta = pd.read_csv("beta.csv")["Beta"]
    # alpha = np.random.normal(0.9928183758944317, 0.06182251581878024, sparse_matrix.shape[1])
    # two_param_irt(theta,beta,sparse_matrix, val_data, test_data, 0.001, 143)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    #####################################################################
    # Implement part (c)                                                #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def data_prep():
    """
        Prepare data-set for XG Boost modelling
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


# def irt_impute(theta, beta):
#     matrix = load_train_sparse("../data").toarray()
#     train_data = pd.DataFrame(load_train_csv("../data"))
#     for i, q in enumerate(train_data["question_id"]):
#         u = train_data["user_id"][i]
#         x = (theta[u] - beta[q]).sum()
#         p_a = sigmoid(x)
#         pred.append(p_a >= 0.5)
#     for i in range(len(data["user_id"])):
#         cur_user_id = data["user_id"][i]
#         cur_question_id = data["question_id"][i]
#         if is.nan(matrix[cur_user_id, cur_question_id]):
#


if __name__ == "__main__":
    main()
    # data_loader()
    # analyse_privdata()
