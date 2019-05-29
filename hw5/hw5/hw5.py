from numpy import count_nonzero, logical_and, logical_or, concatenate, mean, array_split, poly1d, polyfit, array, where, delete, copy, argmax
from numpy.random import permutation
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from itertools import product


SVM_DEFAULT_DEGREE = 3
SVM_DEFAULT_GAMMA = 'auto'
SVM_DEFAULT_C = 1.0
ALPHA = 1.5


def prepare_data(data, labels, max_count=None, train_ratio=0.8):
    """
    :param data: a numpy array with the features dataset
    :param labels:  a numpy array with the labels
    :param max_count: max amout of samples to work on. can be used for testing
    :param train_ratio: ratio of samples used for train
    :return: train_data: a numpy array with the features dataset - for train
             train_labels: a numpy array with the labels - for train
             test_data: a numpy array with the features dataset - for test
             test_labels: a numpy array with the features dataset - for test
    """
    if max_count:
        data = data[:max_count]
        labels = labels[:max_count]

    shuffled_set = concatenate([data, labels.reshape(len(labels), 1)], axis=1)
    shuffled_set = permutation(shuffled_set)



    train_size = round(len(shuffled_set)*train_ratio)
    features_size = data.shape[1]
    train_data = shuffled_set[:train_size, :features_size]
    train_labels = shuffled_set[:train_size, -1:].flatten()
    test_data = shuffled_set[train_size:, :features_size]
    test_labels = shuffled_set[train_size:, -1:].flatten()

    return train_data, train_labels, test_data, test_labels


def get_stats(prediction, labels):
    """
    :param prediction: a numpy array with the prediction of the model
    :param labels: a numpy array with the target values (labels)
    :return: tpr: true positive rate
             fpr: false positive rate
             accuracy: accuracy of the model given the predictions
    """
    positives = count_nonzero(labels)
    negatives = len(labels) - positives
    tpr = 1 if positives == 0 else (count_nonzero(where(logical_and(labels == 1, prediction == labels), 1, 0)) / positives)
    fpr = 0 if negatives == 0 else (count_nonzero(where(logical_and(labels == 0, prediction != labels), 1, 0)) / negatives)
    accuracy = 1 - (count_nonzero(prediction-labels) / len(prediction))

    return tpr, fpr, accuracy


def get_k_fold_stats(folds_array, labels_array, clf):
    """
    :param folds_array: a k-folds arrays based on a dataset with M features and N samples
    :param labels_array: a k-folds labels array based on the same dataset
    :param clf: the configured SVC learner
    :return: mean(tpr), mean(fpr), mean(accuracy) - means across all folds
    """
    tpr = []
    fpr = []
    accuracy = []
    for i in range(0,len(folds_array)):
        folds_copy = copy(folds_array)
        labels_copy = copy(labels_array)
        test_set = folds_copy[i];
        test_labels = labels_copy[i];
        train_set = concatenate(delete(folds_copy, i, axis=0))
        train_labels = concatenate(delete(labels_copy, i, axis=0))
        clf.fit(train_set, train_labels)
        prediction = clf.predict(test_set)
        test_tpr, test_fpr, test_acc = get_stats(prediction, test_labels)
        #print(test_tpr, test_fpr, test_acc)
        tpr.append(test_tpr)
        fpr.append(test_fpr)
        accuracy.append(test_acc)

    return mean(tpr), mean(fpr), mean(accuracy)


def compare_svms(data_array,
                 labels_array,
                 folds_count,
                 kernels_list=('poly', 'poly', 'poly', 'rbf', 'rbf', 'rbf',),
                 kernel_params=({'degree': 2}, {'degree': 3}, {'degree': 4}, {'gamma': 0.005}, {'gamma': 0.05}, {'gamma': 0.5},)):
    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :param kernels_list: a list of strings defining the SVM kernels
    :param kernel_params: a dictionary with kernel parameters - degree, gamma, c
    :return: svm_df: a dataframe containing the results as described below
    """
    svm_df = pd.DataFrame()
    svm_df['kernel'] = kernels_list
    svm_df['kernel_params'] = kernel_params

    folded_data = array_split(data_array, folds_count)
    folded_labels = array_split(labels_array,  folds_count)
    tpr_list = []
    fpr_list = []
    acc_list = []

    for i in range(0, len(kernels_list)):
        clf = SVC(kernel=kernels_list[i], **kernel_params[i])
        tpr, fpr, acc = get_k_fold_stats(folded_data, folded_labels, clf)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        acc_list.append(acc)

    svm_df['tpr'] = tpr_list
    svm_df['fpr'] = fpr_list
    svm_df['accuracy'] = acc_list

    return svm_df


def get_most_accurate_kernel(res):
    """
    :return: integer representing the row number of the most accurate kernel
    """

    best_kernel = argmax(res)
    return best_kernel


def get_kernel_with_highest_score(res):
    """
    :return: integer representing the row number of the kernel with the highest score
    """
    best_kernel = argmax(res)
    return best_kernel


def plot_roc_curve_with_score(df, alpha_slope=1.5):
    """
    :param df: a dataframe containing the results of compare_svms
    :param alpha_slope: alpha parameter for plotting the linear score line
    :return:
    """
    curve = array([df.fpr.tolist(), df.tpr.tolist()])
    curve.sort()
    curve = curve.transpose()
    curve = concatenate([array([0, 0]).reshape(1, 2), curve, array([1, 1]).reshape(1, 2)])

    best_kernel = get_kernel_with_highest_score(df["score"])
    b = df.iloc[best_kernel]["tpr"] - alpha_slope*df.iloc[best_kernel]["fpr"]
    p2 = alpha_slope + b
    plt.figure()
    lw = 2
    plt.plot(curve[:, 0], curve[:, 1], color='darkorange',
             lw=lw)
    plt.plot([0, 1], [b,p2], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def evaluate_c_param(data_array, labels_array, folds_count, best_kernel):
    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :return: res: a dataframe containing the results for the different c values. columns similar to `compare_svms`
    """

    res = pd.DataFrame()
    i = [1, 0,- 1, -2, -3, -4]
    j = [3, 2, 1]
    c_list = []
    tpr_list = []
    fpr_list = []
    acc_list = []
    folded_data = array_split(data_array, folds_count)
    folded_labels = array_split(labels_array,  folds_count)

    def compute_c (pair):
        return (10**pair[0]) * (pair[1]/3)
    for pair in list(product(i,j)):
        c_value = compute_c(pair)
        clf = SVC(C=c_value, kernel=best_kernel["kernel"], **best_kernel["kernel_params"])
        tpr, fpr, acc = get_k_fold_stats(folded_data, folded_labels, clf)
        c_list.append(c_value)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        acc_list.append(acc)

    res["c_value"] = c_list
    res['tpr'] = tpr_list
    res['fpr'] = fpr_list
    res['accuracy'] = acc_list

    return res


def get_test_set_performance(train_data, train_labels, test_data, test_labels, best_kernel):
    """
    :param train_data: a numpy array with the features dataset - train
    :param train_labels: a numpy array with the labels - train

    :param test_data: a numpy array with the features dataset - test
    :param test_labels: a numpy array with the labels - test
    :return: kernel_type: the chosen kernel type (either 'poly' or 'rbf')
             kernel_params: a dictionary with the chosen kernel's parameters - c value, gamma or degree
             clf: the SVM leaner that was built from the parameters
             tpr: tpr on the test dataset
             fpr: fpr on the test dataset
             accuracy: accuracy of the model on the test dataset
    """

    clf = SVC(C=best_kernel["c_value"], kernel=best_kernel["kernel"], **best_kernel["kernel_params"])
    kernel_type = best_kernel["kernel"]
    kernel_params = {"C":best_kernel["c_value"],  **best_kernel["kernel_params"]}
    clf.fit(train_data, train_labels)
    prediction = clf.predict(test_data)

    tpr, fpr, accuracy = get_stats(prediction,test_labels)

    return kernel_type, kernel_params, clf, tpr, fpr, accuracy
