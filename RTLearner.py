import numpy as np

def make_leaf(y):
    node = np.empty((1,4), dtype=float)
    node[0,0] = -1
    node[0,1] = y
    node[0,2] = np.nan
    node[0,3] = np.nan
    return node

def make_root(feature_idx, split_val, left_rows):
    node = np.empty((1,4), dtype=float)
    node[0,0] = feature_idx
    node[0,1] = split_val
    node[0,2] = 1
    node[0,3] = left_rows + 1
    return node

class RTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
        """
        Constructor method
        """
        self.verbose = verbose
        self.leaf_size = leaf_size
        self.tree = None

    def add_evidence(self, train_x, train_y):
        self.tree = self.build_tree(train_x, train_y)
        #print("tree shape: ", self.tree.shape)

    def build_tree(self, train_x, train_y):
        if train_x.shape[0] <= self.leaf_size:
            #return np.array([[-1,np.nanmean(train_y),np.nan,np.nan]])
            return make_leaf(np.nanmean(train_y))
        if np.all(train_y == train_y[0]):
            #return np.array([[-1,train_y[0],np.nan,np.nan]])
            return make_leaf(train_y[0])
        else:
            # calculate correlations between factors and y values and take the factor with the highest correlation
            best_factor = np.random.randint(0, train_x.shape[1])
        splitval = np.nanmedian(train_x[:, best_factor])
        if np.isnan(splitval):
            #return np.array([[-1, np.nanmean(train_y), np.nan, np.nan]])
            return make_leaf(np.nanmean(train_y))
        left_vals = train_x[:, best_factor] <= splitval
        right_vals = train_x[:, best_factor] > splitval
        left_count = np.sum(left_vals)
        right_count = np.sum(right_vals)
        if left_count == 0 or right_count == 0:
            #return np.array([[-1,np.nanmean(train_y),np.nan,np.nan]])
            return make_leaf(np.nanmean(train_y))
        lefttree = self.build_tree(train_x[left_vals], train_y[left_vals])
        righttree = self.build_tree(train_x[right_vals], train_y[right_vals])
        #root = np.array([[best_factor, splitval, 1, lefttree.shape[0] + 1]])
        root = make_root(best_factor, splitval, lefttree.shape[0])
        return np.vstack((root, lefttree, righttree))

    def query(self, test_x):
        pred = np.zeros(test_x.shape[0])
        for i in range(test_x.shape[0]):
            node = 0
            while True:
                feature = int(self.tree[node,0])
                if feature == -1:
                    pred[i] = self.tree[node,1]
                    break
                splitval = self.tree[node,1]
                if test_x[i, feature] <= splitval:
                    node = node + int(self.tree[node,2])

                else:
                    node = node + int(self.tree[node,3])
        return pred
