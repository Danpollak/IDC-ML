import numpy as np
np.random.seed(42)

####################################################################################################
#                                            Part A
####################################################################################################

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """
        self.dataset = dataset
        self.class_value = class_value
        class_rows = dataset[dataset[:, -1] == class_value]
        self.param_dict = []
        count = np.shape(class_rows)[0]
        for param in range(0, (np.shape(class_rows)[1]-1)):
            mean = np.sum(class_rows[:, param])/count
            mu = np.sqrt(np.sum((class_rows[:, param]-mean)*(class_rows[:, param]-mean))/count)
            self.param_dict.insert(param, [mean, mu])
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        class_column = self.dataset[:, -1]
        freq = np.count_nonzero(class_column == self.class_value)
        count = np.shape(class_column)[0]
        return freq/count
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood = 1
        for param in range(0, np.shape(x)[0]):
            likelihood *= normal_pdf(x[param], self.param_dict[param][0], self.param_dict[param][1])
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_prior()*self.get_instance_likelihood(x)


class MultiNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """
        self.dataset = dataset
        self.class_value = class_value
        class_rows = dataset[dataset[:,-1] == class_value]
        self.mean = []
        count = np.shape(class_rows)[0]
        for param in range(0,(np.shape(class_rows)[1]-1)):
            self.mean.append(np.sum(class_rows[:, param])/count)

        self.cov = np.cov(class_rows[:, 0:-1], rowvar=0)

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        class_column = self.dataset[:, -1]
        freq = np.count_nonzero(class_column == self.class_value)
        count = np.shape(class_column)[0]
        return freq / count
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        return multi_normal_pdf(x, self.mean, self.cov)
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_prior()*self.get_instance_likelihood(x)
    

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    pow = (x-mean)*(x-mean) /(2*std*std)
    numerator = np.power(np.e, -pow)
    denominator = np.sqrt(2*np.pi*std*std)

    return numerator/denominator


def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variante normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """

    pow_exp = -0.5*np.dot(np.dot(np.transpose(x - mean), np.linalg.inv(cov)), x - mean)
    numerator = np.power(np.e, pow_exp)
    denominator = np.sqrt(np.power(2 * np.pi, np.shape(cov)[0]) * np.linalg.det(cov))

    return numerator/denominator



####################################################################################################
#                                            Part B
####################################################################################################
EPSILLON = 1e-6 # == 0.000001 It could happen that a certain value will only occur in the test set.
                # In case such a thing occur the probability for that value will EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with la place smoothing.
        
        Input
        - dataset: The dataset from which to compute the probabilites (Numpy Array).
        - class_value : Compute the relevant parameters only for instances from the given class.
        """
        self.dataset = dataset
        self.class_value = class_value
        self.class_rows = dataset[dataset[:, -1] == class_value]
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        class_column = self.dataset[:, -1]
        freq = np.count_nonzero(class_column == self.class_value)
        count = np.shape(class_column)[0]
        return freq / count
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """

        likelihood = 1
        for param in range(0, np.shape(x)[0]):
            n_ij = self.class_rows[self.class_rows[:, param] == x[param]]
            n_ij = np.shape(n_ij)[0] if np.shape(n_ij)[0] > 0 else EPSILLON
            n_i = np.shape(self.class_rows)[0]
            unq = np.shape(np.unique(self.dataset[:, param]))[0]
            likelihood *= ((n_ij + 1) / (n_i+unq))

        return likelihood

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_prior()*self.get_instance_likelihood(x)

    
####################################################################################################
#                                            General
####################################################################################################            
class MAPClassifier():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a postreiori classifier. 
        This class will hold 2 class distribution, one for class 0 and one for class 1, and will predicit and instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
        
        Input
            - An instance to predict.
            
        Output
            - 0 if the posterior probability of class 0 is higher 1 otherwise.
        """
        if self.ccd0.get_instance_posterior(x) > self.ccd1.get_instance_posterior(x):
            return 0
        else:
            return 1

    
def compute_accuracy(testset, map_classifier):
    """
    Compute the accuracy of a given a testset and using a map classifier object.
    
    Input
        - testset: The test for which to compute the accuracy (Numpy array).
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / #testset size
    """
    correct = 0
    size = np.shape(testset)[0]
    for row in testset:
        prediction = map_classifier.predict(row[:-1])
        if prediction == row[-1]:
            correct += 1
    return correct/size
