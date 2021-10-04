import numpy as np
import function as f

# load data and split into training and test sets
def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data('mnist.npz')

x_train = x_train.astype(float)
y_train = y_train.astype(float)
x_test = x_test.astype(float)
y_test = y_test.astype(float)

# note x_train is a numpy array with size (60000, 28 28)
print('The shape of x_train is:', np.shape(x_train))
print('The shape of x_test is:', np.shape(x_test))

# for this demo, only use the first 2000 training examples
x_train = x_train[0:2000, :, :]
y_train = y_train[0:2000]
# for this demo, only use the first 1000 test examples
x_test = x_test[0:1000, :, :]
y_test = y_test[0:1000]

# display the first 9 training examples
f.imgDisplay(x_train[:9], y_train[:9], 3, 3)

# test one-nearest-neighbor algorithm and compute empirical risk
f.testOneNearestNeighbor(x_train, y_train, x_test, y_test)

# test KNN algorithm and compute empirical risk
f.testKNN(x_train, y_train, x_test, y_test, 10)





