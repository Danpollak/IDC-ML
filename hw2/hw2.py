import numpy as np
np.random.seed(42)

chi_table = {0.01  : 6.635,
             0.005 : 7.879,
             0.001 : 10.828,
             0.0005 : 12.116,
             0.0001 : 15.140,
             0.00001: 19.511}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the gini impurity of the dataset.    
    """
    gini = 0.0
    # extract the label column dataset
    label_col = data[:, -1]
    s_size = np.shape(label_col)[0]
    # get frequency array
    _, freq = np.unique(label_col, return_counts=True)
    for el in freq:
        gini += (el/s_size)*(el/s_size)

    return 1 - gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the entropy of the dataset.    
    """
    entropy = 0.0

    # extract the label column dataset
    label_col = data[:, -1]

    # get frequency array
    _, freq = np.unique(label_col, return_counts=True)
    for el in freq:
        p = el / np.shape(label_col)[0]
        entropy += p*np.log2(1/p)

    return entropy


def split(arr, cond):
    return [arr[cond], arr[~cond]]


def impurity_gain(data, attr, threshold, impurity):

    # get impurity value before splitting to subsets
    before_split = impurity(data)

    # split into 2 subsets by threshold
    set_a, set_b = split(data, data[:, attr] < threshold)

    # calculate the weighted impurity value of the split subsets (goodness of split)
    after_split = compute_set_impurity(data, set_a, set_b, impurity)

    # return the difference between the split impurity values (bigger means better)
    return before_split - after_split


def compute_set_impurity(org_set, set_a, set_b, impurity):
    s_size = np.shape(org_set[:, -1])[0]
    return (np.shape(set_a[:, -1])[0]/s_size)*impurity(set_a) + (np.shape(set_b[:, -1])[0]/s_size)*impurity(set_b)

def avg_array(data, attr):

    # get all unique values
    arr = np.unique(data[:, attr])

    # shift all values to the left
    arr_shift_left = np.roll(arr, -1)

    # add each value to its neighbor on its left, average them and drop the rightmost value.
    avg_arr = ((arr + arr_shift_left) / 2)[:-1].copy()

    return avg_arr


def find_best_attribute(data, impurity):

    best_attr = (None, None)
    best_gain = float("-inf")
    # choose
    for attr in range(np.shape(data)[1]-2):
        avg_arr = avg_array(data, attr)
        for avg_val in avg_arr:
            if impurity_gain(data,attr,avg_val,impurity) > best_gain:
                best_attr = (attr, avg_val)
                best_gain = impurity_gain(data,attr,avg_val,impurity)

    return best_attr


def calc_chi_square(data, set_a, set_b):

    chi_square = 0.0

    # extract all the number of instances from all sets
    total_instances = np.shape(data)[0]
    zero_instances, one_instances = get_instance_freq(data)
    a_zero_instances, a_one_instances = get_instance_freq(set_a)
    b_zero_instances, b_one_instances = get_instance_freq(set_b)

    chance_for_zero = zero_instances/total_instances # P(Y=0)
    chance_for_one = one_instances/total_instances # P(Y=1)

    # compute subset a
    d_zero = np.shape(set_a)[0]
    e_zero = d_zero * chance_for_zero
    e_one = d_zero * chance_for_one
    chi_square += ((a_zero_instances - e_zero) * (a_zero_instances - e_zero) / e_zero)
    chi_square += ((a_one_instances - e_one) * (a_one_instances - e_one) / e_one)

    # compute subset b
    d_zero = np.shape(set_b)[0]
    e_zero = d_zero * chance_for_zero
    e_one = d_zero * chance_for_one
    chi_square += ((b_zero_instances - e_zero) * (b_zero_instances - e_zero) / e_zero)
    chi_square += ((b_one_instances - e_one) * (b_one_instances - e_one) / e_one)

    return chi_square


def get_instance_freq(data):

    unique, freq = np.unique(data[:, -1], return_counts=True)
    if np.shape(unique)[0] < 2:
        if unique[0] == 0:
            return freq[0], 0.0
        return 0.0, freq[0]
    return freq[0], freq[1]

def copy_tree(node):
    new_node = DecisionNode(node.feature, node.value, node.chi)
    if node.pure:
        result, amount = node.get_result()
        new_node.set_leaf(result, amount)
    if len(node.children) > 0:
        new_node.add_child(copy_tree(node.children[0]))
        new_node.add_child(copy_tree(node.children[1]))
    return new_node


class DecisionNode:

    # This class will hold everything you require to construct a decision tree.
    # The structure of this class is up to you. However, you need to support basic 
    # functionality as described in the notebook. It is highly recommended that you 
    # first read and understand the entire exercise before diving into this class.
    
    def __init__(self, feature, value, chi=None):
        self.feature = feature # column index of criteria being tested
        self.value = value # value necessary to get a true result
        self.children = []
        self.result = None
        self.pure = False
        self.amount = 0
        if chi is None:
            self.chi = 1
        else:
            self.chi = chi
        
    def add_child(self, node):
        self.children.append(node)

    def set_node_values(self, feature, value):
        self.feature = feature
        self.value = value

    def set_leaf(self, result, amount):
        self.result = result
        self.amount = amount
        self.pure = True

    def get_attr_value(self):
        return self.feature, self.value

    def get_all_children(self):
        return self.children

    def get_left_child(self):
        if len(self.children) > 1:
            return self.children[0]
        return None

    def get_right_child(self):
        if len(self.children) > 1:
            return self.children[1]
        return None

    def get_result(self):
        return self.result, self.amount

    def remove_children(self):
        self.children = []


def build_tree(data, impurity, chi_value=1):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure. 

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.

    Output: the root node of the tree.
    """
    root = DecisionNode(None, None, chi_value)
    current_node = None

    # the queue holds tuples of unset nodes and data subsets
    queue = [(root, data)]

    while len(queue) != 0:
        # remove node from queue
        current_node = queue.pop()
        node = current_node[0]
        data_subset = current_node[1]

        # if the node is not pure
        if not current_node[0].pure:

            # if the node has single row dataset

            if np.shape(data_subset)[0] == 1 or impurity(data_subset) == 0:
                node.set_leaf(data_subset[0, -1], np.shape(data_subset)[0])

            else:

                # find best attr and threshold, then set it as the node values
                best_attr, threshold = find_best_attribute(data_subset, impurity)
                node.set_node_values(best_attr, threshold)

                # create new children nodes
                left_node = DecisionNode(None, None, chi_value)
                right_node = DecisionNode(None, None, chi_value)

                # check if the split is perfect
                left_set, right_set = split(data_subset, data_subset[:, best_attr] < threshold)
                is_pure = compute_set_impurity(data_subset, left_set, right_set, impurity) == 0

                # check for chi square values. if not good enough, create leaf
                chi_square_val = calc_chi_square(data_subset, left_set, right_set)
                if chi_value < 1 and chi_square_val < chi_table[node.chi]:
                    node.set_leaf(data_subset[0, -1], np.shape(data_subset)[0])

                # if the split is perfect, create two leaves
                elif is_pure:
                    left_node.set_leaf(left_set[0, -1], np.shape(left_set)[0])
                    right_node.set_leaf(right_set[0, -1], np.shape(right_set)[0])

                # else, add the two nodes into the queue
                else:
                    queue.append((left_node, left_set))
                    queue.append((right_node, right_set))

                # finally, add the new nodes as children
                node.add_child(left_node)
                node.add_child(right_node)

    return root


def predict(node, instance):
    """
    Predict a given instance using the decision tree

    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.

    Output: the prediction of the instance.
    """
    pred = None
    current_node = node
    while not current_node.pure:
        attr, threshold = current_node.get_attr_value()

        if instance[attr] < threshold:
            current_node = current_node.get_left_child()

        else:
            current_node = current_node.get_right_child()

    pred,_ = current_node.get_result()
    return pred

def calc_accuracy(node, dataset):
    """
    calculate the accuracy starting from some node of the decision tree using
    the given dataset.

    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0.0
    for row in dataset:
        pred = predict(node, row)
        if pred == row[-1]:
            accuracy += 1

    accuracy /= np.shape(dataset)[0]
    return accuracy*100


def post_prune_tree(root, dataset):

    best_tree = None
    best_accu = float('-inf')

    best_tree, best_accu = find_redundant_leaf(root, root, dataset)
    return best_tree, best_accu


def find_redundant_leaf(node, root, dataset):

    # check if current is a parent to a leaf
    best_tree = None
    best_accu = float('-inf')

    if node.pure:
        return None, best_accu

    if node.get_left_child().pure and node.get_right_child().pure:
        root_copy = copy_tree(root)
        node_copy = find_node(root_copy, node)
        leafify(node_copy)
        best_tree = root_copy
        best_accu = calc_accuracy(best_tree, dataset)

    left_best_tree, left_best_accu = find_redundant_leaf(node.get_left_child(), root, dataset)
    right_best_tree, right_best_accu = find_redundant_leaf(node.get_right_child(), root, dataset)
    if best_accu > left_best_accu and best_accu > right_best_accu:
        return best_tree, best_accu
    elif left_best_accu > best_accu and left_best_accu > right_best_accu:
        return left_best_tree, left_best_accu
    else:
        return right_best_tree, right_best_accu


def leafify(node):
        zero_amount = 0
        one_amount = 0

        if node.get_left_child().result > 0:
            one_amount += node.get_left_child().amount
        else:
            zero_amount += node.get_left_child().amount

        if node.get_right_child().result > 0:
            one_amount += node.get_right_child().amount
        else:
            zero_amount += node.get_right_child().amount

        leaf_val = 1.0 if one_amount > zero_amount else 0.0
        node.set_leaf(leaf_val, zero_amount + one_amount)
        node.remove_children()


def tree_size(node):
        if node.pure:
            return 0
        else:
            return tree_size(node.get_left_child()) + 1 + tree_size(node.get_right_child())


def find_node(node, search):

    if node.pure:
        return False

    if node.value == search.value and node.feature == search.feature:
        return node
    return find_node(node.get_left_child(), search) or find_node(node.get_right_child(), search)


def print_tree(node, level=0):
    '''
    prints the tree according to the example in the notebook

	Input:
	- node: a node in the decision tree

	This function has no return value
	'''

    print(print_node(node,level))
    if not node.pure:
        print_tree(node.get_left_child(), level + 1)
        print_tree(node.get_right_child(), level + 1)


def print_node(node, level):

    node_string="   "

    for it in range(0, level):
        node_string += "    "
    if node.pure:
        result, amount = node.get_result()
        node_string+="leaf: [{" + str(result) + ": " + str(amount) + "}]"
        return node_string
    attr, value = node.get_attr_value()
    node_string+="[X" + str(attr) + " <= " + str(value) + "],"
    return node_string
