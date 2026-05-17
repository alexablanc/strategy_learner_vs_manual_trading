import numpy as np

class BagLearner(object):

    def __init__(self, learner, kwargs, bags=20, boost=False, verbose=False):
        """
        Constructor method
        """
        self.kwargs = kwargs
        self.learner = learner
        self.bags = bags
        self.verbose = verbose
        self.learners = []
        for i in range(bags):
            self.learners.append(learner(**kwargs))

    def add_evidence(self, train_x, train_y):
        for learner in self.learners:
            indices = np.random.choice(train_x.shape[0], size=train_x.shape[0], replace=True)
            learner.add_evidence(train_x[indices], train_y[indices])

    def query(self, test_x):
        pred = np.array([learner.query(test_x) for learner in self.learners])
        return np.mean(pred, axis=0)

