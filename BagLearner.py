import numpy as np
from scipy import stats


class BagLearner(object):
    """
    This is a Bag Learner.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

    def __init__(self, learner, kwargs, bags = 20, boost = False, verbose=False):
        """
        Constructor method
        """
        self.learners = [learner(**kwargs) for i in range(bags)]
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose


    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        n = data_x.shape[0]

        # build and save the model
        for i in range(self.bags):
            random_rows = np.random.choice(np.arange(n), n, replace=True)
            self.learners[i].add_evidence(data_x[random_rows,:],data_y[random_rows])


    def query(self, points):
        """
        Estimate a set of test points given the model we built.
        """
        results = np.zeros((points.shape[0], self.bags))
        for i in range(self.bags):
            results[:,i] = self.learners[i].query(points)
        if self.verbose:
            print(results)
        #results = results.sum(axis=1)
        #results[results >= 1] = 1
        #results[results <= -1] = -1

        return stats.mode(results, axis=1)[0]


if __name__ == "__main__":
    print("This is Bag Learner'")
