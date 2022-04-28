import numpy as np
from scipy import stats


class RTLearner(object):
    """
    This is a Random Tree Learner. It is implemented correctly.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

    def __init__(self, leaf_size=1, verbose=False):
        """
        Constructor method
        """
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "lsu63"

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """

        # concatenate x and y
        combined_data = np.empty((data_x.shape[0], (data_x.shape[1] + 1)))
        combined_data[:, :-1] = data_x
        combined_data[:, -1] = data_y
        # build and save the model
        self.tree = self.build_tree(combined_data)

    def build_tree(self, data):
        if (data.shape[0] == 1) or (np.all(data[:, -1] == data[:, -1][0])):
            return np.array([-1, data[:, -1][0], np.NaN, np.NaN])

        if data.shape[0] <= self.leaf_size:
            return np.array([-1, stats.mode(data[:, -1])[0][0], np.NaN, np.NaN])

        i = np.random.randint(0,data.shape[1]-1)
        split_val = np.median(data[:, i])
        if (np.all(data[:, i] <= split_val)) or (np.all(data[:, i] > split_val)):
            return np.array([-1, stats.mode(data[:, -1])[0][0], np.NaN, np.NaN])

        else:
            left_tree = self.build_tree(data[data[:, i] <= split_val])
            right_tree = self.build_tree(data[data[:, i] > split_val])
            if left_tree.ndim == 1:
                root = np.array([i, split_val, 1, 2])
            else:
                root = np.array([i, split_val, 1, left_tree.shape[0] + 1])
            return np.vstack((root, left_tree, right_tree))

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        num_of_points = points.shape[0]
        result = np.zeros(shape=(num_of_points,))
        for i in range(num_of_points):
            result[i] = self.search_tree(points[i, :])
        if self.verbose:
            print(result)
        return result

    def search_tree(self, x):
        """
        Estimate predicted value for one data point given the model we built.
        """
        if self.tree[0, 0] == -1:
            return self.tree[0, 1]
        else:
            leaf = False
            i = 0
            while not leaf:
                split_feature = int(self.tree[i, 0])
                split_val = self.tree[i, 1]

                if x[split_feature] <= split_val:
                    i += int(self.tree[i, 2])
                    if int(self.tree[i, 0]) == -1:
                        leaf = True
                else:
                    i += int(self.tree[i, 3])
                    if int(self.tree[i, 0]) == -1:
                        leaf = True
            return self.tree[i, 1]

def author():
    return 'lsu63'

if __name__ == "__main__":
    print("This is Random Tree Learner")
