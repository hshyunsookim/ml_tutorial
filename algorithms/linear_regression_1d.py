import numpy as np
import matplotlib.pyplot as plt

# settings
plot = True
alpha = 0.01
iter = 1500

# define helper functions
def computeCost(X, y, theta):
    y_hat = np.dot(X,theta)
    J = 0.5*np.mean(np.square(y_hat - y))
    return J

def computeGradient(X, y, theta):
    y_hat = np.dot(X,theta)
    dTheta = np.mean(np.multiply((y_hat - y), X), axis = 0)
    return np.reshape(dTheta, (2,1))

if __name__ == "__main__":

    # load comma separated dataset, skip first-line header
    data = np.genfromtxt('../dataset/ex1data1.txt', delimiter=',', skip_header = 1)
    x = data[:,0, None]
    y = data[:,1, None]
    N = y.size
    print "dataset contains", N, "entries"

    # augment x by adding a column of 1's  (x <-- [1, x])
    X = np.c_[np.ones((N,1)), x]
    D = X[1,:].size
    print "dataset contains", D, "features, including one additional term"

    # initialize parameters
    theta = np.zeros((D,1))

    # repeat gradient descent
    for i in range(iter):

        # compute cost
        if i%100 == 0:
            print("cost at iter #%d = %.2f" % (i, computeCost(X,y,theta)))

        # compute gradient descent step
        dTheta = computeGradient(X,y,theta)

        # update parameters
        theta = theta - alpha*dTheta

    # plot regression result
    if plot:
        plt.plot(x,y, 'rx')
        plt.plot(x, np.dot(X,theta), 'b-')
        plt.xlabel('Population of City in 10,000s')
        plt.ylabel('Profits in $10,000s')
        plt.show()
