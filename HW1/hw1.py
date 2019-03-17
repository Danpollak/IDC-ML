import numpy as np
import itertools
np.random.seed(42)

def preprocess(X, y):
    """
    Perform mean normalization on the features and divide the true labels by
    the range of the column. 

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels.

    Returns a two vales:
    - X: The mean normalized inputs.
    - y: The scaled labels.
    """
    newx = np.empty_like(X).astype(float);
    newy = np.empty_like(y).astype(float);
    for i in range(0,np.size(X,0)):
        maxVal = np.amax(X[i])
        minVal = np.amin(X[i])
        avg = np.average(X[i])
        def normvalues(i):
            return (i - avg) / (maxVal - minVal)

        vnorm = np.vectorize(normvalues)
        newx[i:] = vnorm(X[i:])

    maxVal = np.amax(y)
    minVal = np.amin(y)
    avg = np.average(y)

    def normvalues(i):
        return (i - avg) / (maxVal - minVal)

    vnorm = np.vectorize(normvalues)
    newy = vnorm(y)


    X=newx
    y=newy
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X, y

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation’s actual and
    predicted values for linear regression.  

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).
    - theta: The parameters (weights) of the model being learned.

    Returns a single value:
    - J: the cost associated with the current set of parameters (single number).
    """

    coeff = 1.0/(2 * X.shape[0])
    diff = np.matmul(X, theta) - y
    diff = np.power(diff, 2)
    sum = np.sum(diff)
    J = coeff * sum
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent. Gradient descent
    is an optimization algorithm used to minimize some (loss) function by 
    iteratively moving in the direction of steepest descent as defined by the
    negative of the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns two values:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    J_history = []  # Use a python list to save cost in every iteration
    coeff = alpha/X.shape[0];
    for it in range(0, num_iters):
        diffX = np.matmul(X,theta) - y
        sum = np.sum(diffX)
        temp = [0,0]
        temp[0] = theta[0] - coeff * np.sum(diffX)
        temp[1] = theta[1] - coeff * np.matmul(diffX, X[:, 1])
        theta = temp
        cost = compute_cost(X, y, theta)
        J_history.append(cost)
    return theta, J_history

def pinv(X, y):
    """
    Calculate the optimal values of the parameters using the pseudoinverse
    approach as you saw in class.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).

    Returns two values:
    - theta: The optimal parameters of your model.

    ########## DO NOT USE numpy.pinv ##############
    """
    XT = np.transpose(X)
    pinv = np.matmul(np.linalg.inv(np.matmul(XT, X)), XT)
    pinv_theta = np.matmul(pinv, y)
    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model, but stop the learning process once
    the improvement of the loss value is smaller than 1e-8. This function is
    very similar to the gradient descent function you already implemented.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns two values:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    J_history = []  # Use a python list to save cost in every iteration
    J_history.append(compute_cost(X, y, theta))
    coeff = alpha/X.shape[0];
    for it in range(0, num_iters):
        diffX = np.matmul(X,theta) - y
        temp = [0,0]
        temp[0] = theta[0] - coeff * np.sum(diffX)
        temp[1] = theta[1] - coeff * np.matmul(diffX, X[:, 1])
        theta = temp
        J_history.append(compute_cost(X, y, theta))
        loss = J_history[it+1]
        prev_loss = J_history[it]
        loss_diff = prev_loss - loss
        if loss_diff < 1e-8:
            print(theta)
            print(it)
            break
    return theta, J_history

def find_best_alpha(X, y, iterations):
    """
    Iterate over provided values of alpha and maintain a python dictionary 
    with alpha as the key and the final loss as the value.

    Input:
    - X: a dataframe that contains all relevant features.

    Returns:
    - alpha_dict: A python dictionary that hold the loss value after training 
    for every value of alpha.
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return alpha_dict

def generate_triplets(X):
    """
    generate all possible sets of three features out of all relevant features
    available from the given dataset X. You might want to use the itertools
    python library.

    Input:
    - X: a dataframe that contains all relevant features.

    Returns:
    - A python list containing all feature triplets.
    """
    
    triplets = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return triplets

def find_best_triplet(df, triplets, alpha, num_iter):
    """
    Iterate over all possible triplets and find the triplet that best 
    minimizes the cost function. For better performance, you should use the 
    efficient implementation of gradient descent. You should first preprocess
    the data and obtain a array containing the columns corresponding to the
    triplet. Don't forget the bias trick.

    Input:
    - df: A dataframe that contains the data
    - triplets: a list of three strings representing three features in X.
    - alpha: The value of the best alpha previously found.
    - num_iters: The number of updates performed.

    Returns:
    - The best triplet as a python list holding the best triplet as strings.
    """
    best_triplet = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return best_triplet
