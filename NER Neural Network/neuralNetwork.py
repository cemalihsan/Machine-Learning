import math

from preprocess import read_data, create_samples, split_data, read_vocab, vector_read
import numpy as np
from scipy.special import expit

# HYPERPARAMETERS
input_size = 50  # size of each word vector
output_size = 2  # number of classes
hidden_layer_size = 100
learning_rate = 0.0001
number_of_epochs = 2
path = "./data"  # use relative path like this

dict_array = {}
vocab_array = []
vector_array = []

W1 = np.random.randn(hidden_layer_size, 3 * input_size) * 0.1
W2 = np.random.randn(hidden_layer_size, output_size) * 0.1
B1 = np.random.randn(hidden_layer_size, 1)
B2 = np.random.randn(output_size, 1)


def vector_to_dict():
    read_vocab()
    vector_read()
    dictionary = dict(zip(vocab_array, vector_array))
    return dictionary


def activation_function(layer):
    value = np.clip(layer, a_min=-745, a_max=710)
    tan = (np.exp(value) - np.exp(-value)) / (np.exp(value) + np.exp(-value))
    return tan


def derivation_of_activation_function(signal):
    tan = activation_function(signal)
    derivation = 1 - (tan ** 2)
    return derivation


def loss_function(true_labels, probabilities):
    p = expit(probabilities)
    Y = true_labels
    Y_hat = p
    cost = (-1/len(Y_hat))*(np.multiply(Y, np.log(Y_hat)) + np.multiply(1-Y, 1-np.log(Y_hat)))
    return cost


# the derivation should be with respect to the output neurons
def derivation_of_loss_function(true_labels, probabilities):
    p = expit(probabilities)
    #loss_output = np.subtract(probabilities, true_labels)
    loss_output = (np.divide(true_labels, p) - np.divide(1 - true_labels, 1 - p))
    return loss_output


# softmax is used to turn activations into probability distribution
def softmax(layer):
    #return 1 / (1 + np.exp(-layer))
    return expit(layer) # in order to solve exp problem scipy provides expit function for sigmoid calculation


def sigmoid_derivative(layer):
    # computing derivative to the Sigmoid function
    result = softmax(layer) * (1 - softmax(layer))
    return result


def forward_pass(data):
    z1 = np.dot(W1, data) + B1 #100x1 matrix
    # print(z1)
    hidden = activation_function(z1)
    #print(hidden.shape[0], hidden.shape[1])
    z2 = np.dot(W2.T, hidden) + B2 #2x1 matrix
    # print(z2.shape[0], z2.shape[1])
    output = softmax(z2)
    # print(output)
    return output, hidden


# should change the strings into word vectors. Should not be effected by the backpropagation
def embedding_layer(samples):
    dict_vector = []
    vec_array = []
    for element in samples:
        if element in dict_array:
            vec_array.append(dict_array[element])
        else:
            vec_array.append(np.random.randn(50))

    for each_element in vec_array:
        for each_vector in each_element:
            dict_vector.append([each_vector])
    vec_dict_array = np.reshape(dict_vector, (len(dict_vector), 1))
    """
    for xx in vec_dict_array:
        if math.isnan(xx):
            print(xx, math.isnan(xx))
    """
    return vec_dict_array


# [hidden_layers] is not an argument, replace it with your desired hidden layers
def backward_pass(input_layer, hidden_layer, output_layer, loss):
    global W1, W2, B1, B2

    # print(input_layer.shape[0], input_layer.shape[1])
    # print(output_layer.shape[0], output_layer.shape[1])
    #print(hidden_layer.shape[0], hidden_layer.shape[1])
    # print(loss.shape[0], loss.shape[1])
    #print(W2.shape[0], W2.shape[1])

    output_delta2 = np.multiply(loss, sigmoid_derivative(output_layer)) #2x1 matrix
    # print(output_delta2.shape[0], output_delta2.shape[1])
    hidden_layer_input = np.add(np.dot(W1, input_layer), B1)
    output_delta1 = np.multiply(hidden_layer, derivation_of_activation_function(hidden_layer_input))#100x1
    #print(output_delta1.shape[0], output_delta1.shape[1])
    # print(sigmoid_derivative(output_layer).shape[0], sigmoid_derivative(output_layer).shape[1])
    derivative_W1 = np.dot(output_delta1, input_layer.T) #100x150
    #print(derivative_W1.shape[0], derivative_W1.shape[1])
    derivative_W2 = np.dot(hidden_layer, output_delta2.T)# 100x2

    W1 = W1 - learning_rate * derivative_W1 #100x150
    W2 = W2 - learning_rate * derivative_W2 #100x2
    B1 = B1 - learning_rate * output_delta1#100x2
    B2 = B2 - learning_rate * output_delta2#2x1

    return 0


def train(train_data, train_labels, valid_data, valid_labels):
    loss_values = []
    for epoch in range(number_of_epochs):
        index = 0

        # for each batch
        for data, labels in zip(train_data, train_labels):
            # Same thing about [hidden_layers] mentioned above is valid here also
            labels = np.reshape(2, 1)
            embedding_array = embedding_layer(data)
            predictions, hidden_layer = forward_pass(embedding_array)
            #print(predictions.shape[0], predictions.shape[1], hidden_layer.shape[0], hidden_layer.shape[1])
            #print(predictions)
            loss_signals = derivation_of_loss_function(labels, predictions)
            # print(loss_signals)
            backward_pass(embedding_array, hidden_layer, predictions, loss_signals)
            loss = loss_function(labels, predictions)

            if index % 20000 == 0:  # at each 20000th sample, we run validation set to see our model's improvements
                accuracy, loss = test(valid_data, valid_labels)
                print("Epoch= " + str(epoch) + ", Coverage= %" + str(
                    100 * (index / len(train_data))) + ", Accuracy= " + str(accuracy) + ", Loss= " + str(loss))
                loss_values.append(loss)
            index += 1

    return loss_values


def test(test_data, test_labels):
    avg_loss = 0
    predictions = []
    labels = []

    # for each batch
    for data, label in zip(test_data, test_labels):
        label = np.reshape(2, 1)
        embedding_data = embedding_layer(data)
        prediction, hidden_layer = forward_pass(embedding_data)
        predictions.append(prediction)
        # print(predictions)
        labels.append(label)
        # l = loss_function(label, prediction)
        avg_loss += np.sum(loss_function(label, prediction))


    # turn predictions into one-hot encoded
    one_hot_predictions = np.zeros(shape=(len(predictions), output_size))
    for i in range(len(predictions)):
        one_hot_predictions[i][np.argmax(predictions[i])] = 1

    predictions = one_hot_predictions
    # print(predictions)
    accuracy_score = accuracy(labels, predictions)

    return accuracy_score, avg_loss / len(test_data)


def accuracy(true_labels, predictions):
    true_pred = 0

    for i in range(len(predictions)):
        if np.argmax(predictions[i]) == np.argmax(true_labels[i]):  # if 1 is in same index with ground truth
            true_pred += 1

    return true_pred / len(predictions)


if __name__ == "__main__":

    # PROCESS THE DATA
    dict_array = vector_to_dict()
    words, labels = read_data(path)
    sentences = create_samples(words, labels)
    train_x, train_y, test_x, test_y = split_data(sentences)

    # creating one-hot vector notation of labels. (Labels are given numeric)
    # [0 1] is PERSON
    # [1 0] is not PERSON
    new_train_y = np.zeros(shape=(len(train_y), output_size))
    new_test_y = np.zeros(shape=(len(test_y), output_size))

    for i in range(len(train_y)):
        new_train_y[i][int(train_y[i])] = 1

    for i in range(len(test_y)):
        new_test_y[i][int(test_y[i])] = 1

    train_y = new_train_y
    test_y = new_test_y

    # Training and validation split. (%80-%20)
    valid_x = np.asarray(train_x[int(0.8 * len(train_x)):-1])
    valid_y = np.asarray(train_y[int(0.8 * len(train_y)):-1])
    train_x = np.asarray(train_x[0:int(0.8 * len(train_x))])
    train_y = np.asarray(train_y[0:int(0.8 * len(train_y))])

    train(train_x, train_y, valid_x, valid_y)
    print("Test Scores:")
    print(test(test_x, test_y))
