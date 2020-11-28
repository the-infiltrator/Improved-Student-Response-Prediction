from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

import matplotlib.pyplot as plt

# Set seed as the date of training and model evaluation
print("seed set")
np.random.seed(20201128)


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)
        self.sigmoid = nn.Sigmoid()

    def get_weight_norm(self):
        """ Return ||W^1|| + ||W^2||.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2)
        h_w_norm = torch.norm(self.h.weight, 2)
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        out = inputs
        out = self.g(out)
        out = self.sigmoid(out)
        out = self.h(out)
        out = self.sigmoid(out)
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch,
          metrics, k):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]
    print(
        f"#########################################################################################\n"
        f"                        FITTING AUTO-ENCODER : λ = {lamb}, k = {k}, α = {lr}         \n"
        f"#########################################################################################\n")
    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.) + lamb / (
                    2 * num_student) * model.get_weight_norm()
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)

        # Update Metrics
        metrics["k"][k]["Training Cost"].append(train_loss)
        metrics["k"][k]["Validation Accuracy"].append(valid_acc)

        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {} \n".format(epoch, train_loss, valid_acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def get_plots(metrics):
    for k in metrics:
        data = metrics[k]
        fig, ax1 = plt.subplots()
        ax1.plot(list(range(len(data["Cost"]))), data["Cost"], color="red")
        ax1.set_xlabel("Epoch, k=" + str(k))
        ax1.set_ylabel("Cost")

        ax2 = ax1.twinx()
        ax2.plot(list(range(len(data["Cost"]))), data["Validation Accuracy"],
                 color="blue")
        ax2.set_xlabel("Epoch, lambda=" + str(k))
        ax2.set_ylabel("Validation Accuracy")
        plt.show()


def gen_tuning_plots(metrics, k_values, lr):
    plt.style.use('ggplot')
    plt.style.use('seaborn-paper')
    plt.subplots(1, 5, figsize=(60, 3.5), tight_layout=True)
    for i in range(len(k_values)):
        data = metrics["k"][k_values[i]]
        plt.subplot(1, 5, i + 1)
        plt.title(f'Accuracy for k = {str(k_values[i])} and α = {lr}')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Validation Accuracy')
        # print(data["Validation Accuracy"])
        plt.plot(np.array(data["Validation Accuracy"]), color="darkcyan", label="Validation Set")
        # plt.plot(evaluation_stats[lamb_vals[i - 1]][0]["Cross-Entropy"], color="firebrick", label="Training Set")
        plt.legend(loc="upper left")
    plt.show()


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    np.random.seed(311)
    num_questions = zero_train_matrix.shape[1]
    #####################################################################
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################

    # Set model hyperparameters.
    k_values = [10, 50, 100, 200, 500]
    num_questions = zero_train_matrix.shape[1]
    metrics = {"k": {}, "lambda": {}}

    # Set optimization hyperparameters.
    lr = 0.021
    num_epoch = 100
    lamb = 0
    for k in k_values:
        model = AutoEncoder(num_question=num_questions, k=k)
        metrics["k"][k] = {"Validation Cost": [], "Training Cost": [], "Validation Accuracy": [], "Test Accuracy": []}
        train(model, lr, lamb, train_matrix, zero_train_matrix,
              valid_data, num_epoch, metrics, k)
        metrics["k"][k]["Test Accuracy"].append(evaluate(model, zero_train_matrix, test_data))
    best_values = [np.max([metrics["k"][k]["Validation Accuracy"]]) for k in k_values]
    best_validation_acc = np.max((best_values))
    best_k = k_values[best_values.index(best_validation_acc)]
    print(
        f'############################################################################################################################\n'
        f'                                       EXPERIMENT COMPLETE,  α = {lr},  Epochs = {num_epoch}, λ = {lamb}\n '
        f'                                    Best  K = {best_k}, Validation Accuracy = {best_validation_acc}\n '
        f'############################################################################################################################\n')

    # Generate side by side accuracy plots for each k
    gen_tuning_plots(metrics, k_values, lr)

    #####################################################################
    # TODO : Use chosen k* -best_k- to train model and plot training and validation #
    # objectives and compute test accuracy                              #
    #####################################################################
    #

    # Tuning Shrinkage Parameter
    # k_star = 10
    # # # Set optimization hyperparameters.
    # lr = 0.021
    # num_epoch = 70
    # lambda_values = [0, 0.001, 0.01, 0.1, 1]
    # for lamb in lambda_values:
    #     model = AutoEncoder(num_question=num_questions, k=k_star)
    #     metrics["lambda"][lamb] = {"Cost": [], "Validation Accuracy": [], "Test Accuracy": []}
    #     train(model, lr, lamb, train_matrix, zero_train_matrix,
    #           valid_data, num_epoch, metrics, k_star)
    #     metrics["lambda"][lamb]["Test Accuracy"].append(evaluate(model, zero_train_matrix, test_data))
    #
    # # get_plots(metrics)
    # best_values = [np.max([metrics["lambda"][lamb]["Validation Accuracy"]]) for lamb in lambda_values]
    # best_validation_acc = np.max((best_values))
    # best_lamb = lambda_values[best_values.index(best_validation_acc)]
    # print(
    #     f'############################################################################################################################\n'
    #     f'                                       EXPERIMENT COMPLETE,  α = {lr},  Epochs = {num_epoch}, k = {k}\n '
    #     f'                Best λ = {best_lamb}, Final Validation Accuracy = {metrics["lambda"][best_lamb]["Validation Accuracy"][-1]}, Final Test Accuracy = {metrics["lambda"][best_lamb]["Test Accuracy"][-1]}\n '
    #     f'############################################################################################################################\n')

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
