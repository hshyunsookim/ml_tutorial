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

def plotContour(theta_min, theta_max):
    theta_range = [np.arange(theta_min[0], theta_max[0], 0.1), \
                   np.arange(theta_min[1], theta_max[1], 0.1)]

    theta1mesh, theta2mesh = np.meshgrid(theta_range[0], theta_range[1])

    Jmesh = np.zeros((theta1mesh.shape[0], theta1mesh.shape[1]))

    for i in range(theta1mesh.shape[0]):
        for j in range(theta1mesh.shape[1]):
            Jmesh[i,j] = computeCost(X,y,[ [theta1mesh[i,j]], [theta2mesh[i,j]] ])

    # levels = [10, 50, 250, 500, 1000, 1500]
    levels = [10, 50, 100, 200, 500, 1000]

    plt.figure()
    CS = plt.contour(theta1mesh, theta2mesh, Jmesh, levels)
    plt.clabel(CS, fontsize=10)
    plt.plot(theta[0], theta[1], 'rx')
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    plt.title('Contour Plot for Cost J')

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

    # print final parameters
    print "final parameters are:", theta[0], theta[1]

    if plot:
        plt.figure()
        plt.plot(x,y, 'rx')
        plt.plot(x, np.dot(X,theta), 'b-')
        plt.xlabel('Population of City in 10,000s')
        plt.ylabel('Profits in $10,000s')

        theta_min = np.concatenate((theta[0]-7, theta[1]-2.5))
        theta_max = np.concatenate((theta[0]+14, theta[1]+4))
        plotContour(theta_min, theta_max)
        plt.show()
