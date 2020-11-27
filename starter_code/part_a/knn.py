from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix.T).T
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def plot_acc(k_set, accuracy_set):
    """
    Plot the accuracy on the validation data as a function of k.

    :param k_set: array of k values
    :param accuracy_set: array of validation accuracy for each k
    :return: None
    """
    plt.plot(k_set, accuracy_set, color='blue', marker='o',
             label='Validation')
    plt.title("Accuracy Vs Number of Nearest Neighbours")
    plt.xlabel('k - Number of Nearest Neighbours')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


def knn_test(matrix, test_data, k, case):
    """
    Report the final test accuracy with the chosen k value for k-Nearest
    Neighbours.

    :param matrix: 2D sparse matrix
    :param test_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    if case == 'user':
        mat = nbrs.fit_transform(matrix)
    else:
        mat = nbrs.fit_transform(matrix.T).T
    test_acc = sparse_matrix_evaluate(test_data, mat)
    print("Final Test Accuracy: {}".format(test_acc))
    return test_acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    accuracy_set_user = []
    k_set = [1, 6, 11, 16, 21, 26]

    for k in k_set:
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        accuracy_set_user.append(acc)

    # plot showing accuracy with the validation set for each k
    plot_acc(k_set, accuracy_set_user)

    # find the k value with best performance
    k_best_index = accuracy_set_user.index(np.max(accuracy_set_user))
    k_best_user = k_set[k_best_index]
    print('k* with highest performance on validation data is, k = {}'
          .format(k_best_user))

    # report final test accuracy
    test_acc_user = knn_test(sparse_matrix, test_data, k_best_user, 'user')

    # item-based collaborative filtering
    accuracy_set_item = []

    for k in k_set:
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        accuracy_set_item.append(acc)

    # plot showing accuracy with the validation set for each k
    plot_acc(k_set, accuracy_set_item)

    # find the k value with best performance
    k_best_index = accuracy_set_item.index(np.max(accuracy_set_item))
    k_best_item = k_set[k_best_index]
    print('k* with highest performance on validation data is, k = {}'
          .format(k_best_item))

    # report final test accuracy
    test_acc_item = knn_test(sparse_matrix, test_data, k_best_item, 'item')
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
