import numpy as np
import matplotlib.pyplot as plt


def imgDisplay(img, lab, rows, cols):
    '''
    Display some samples from dataset in a plot partitioned into rows*cols parts.

    Args:
        img: the x_training data to display.
        lab: the corresponding label to the x_training data to display.
        rows: the number of rows of subplots.
        cols: the number of columns of subplots.
    '''
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img[i], cmap=plt.cm.gray_r)
        plt.xlabel(lab[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()


def oneNearestNeighbor(_x, x_train, y_train):
    '''
    Classifying the data point by finding the 'closest' data point in the training
    set, and using its label as the prediction.

    Args:
        _x: the data point to classify.
        x_train: the training set.
        y_train: the labels of the training set.
    '''
    assert np.shape(_x) == np.shape(x_train[0])  # check to make sure input is a single 28x28 np array
    dis = [np.linalg.norm(_x - each) for each in x_train]
    nearest_neighbor = np.argmin(dis)

    return (y_train[nearest_neighbor], nearest_neighbor)  # return the label of the nearest neighbor, and index


def testOneNearestNeighbor(x_train, y_train, x_test, y_test):
    '''
    Test the oneNearestNeighbor algorithm and compute and print the empirical risk
    by using the misclassification loss and the squared error loss function.

    Args:
        x_train: the training set.
        y_train: the labels of the training set.
        x_test: the test set.
        y_test: the labels of the test set.
    '''
    y_hat = np.array([oneNearestNeighbor(each, x_train, y_train) for each in x_test])
    flag = np.not_equal(y_hat[:, 0], y_test)
    # empirical risk when using the misclassification(0/1) loss
    print('When using the misclassification loss, the empirical risk is:', sum(flag) / len(y_test))
    # empirical risk when using the squared error loss function
    error = np.linalg.norm(y_hat[:, 0] - y_test)
    print('When using the squared error loss function, the empirical risk is:', error ** 2 / len(y_test))


def kNN(_x, x_train, y_train, k):
    '''
    Classifying the data point by finding k 'closest' data points in the training
    set, and using the most commonly occurring label of them as the prediction.

    Args:
        _x: the data point to classify.
        x_train: the training set.
        y_train: the labels of the training set.
        k: the number of 'closest' data points to use.
    '''
    assert np.shape(_x) == np.shape(x_train[0])  # check to make sure input is a single 28x28 np array
    dis = [np.linalg.norm(_x - each) for each in x_train]
    k_nearest_neighbor_index = np.argsort(dis)[:k]
    k_nearest_neighbor = (y_train[k_nearest_neighbor_index]).astype(int)
    y_hat = np.argmax(np.bincount(k_nearest_neighbor))

    return y_hat


def testKNN(x_train, y_train, x_test, y_test, k):
    '''
    Test the KNN algorithm and compute and print the empirical risk
    by using the misclassification loss and the squared error loss function.

    Args:
        x_train: the training set.
        y_train: the labels of the training set.
        x_test: the test set.
        y_test: the labels of the test set.
        k: the number of 'closest' data points to use.
    '''
    y_hat = np.array([kNN(each, x_train, y_train, k) for each in x_test])
    flag = np.not_equal(y_hat, y_test)
    # empirical risk when using the misclassification(0/1) loss
    print('When using the misclassification loss, the empirical risk is:', sum(flag)/len(y_test))
    # empirical risk when using the squared error loss function
    error = np.linalg.norm(y_hat-y_test)
    print('When using the squared error loss function, the empirical risk is:', error**2/len(y_test))

