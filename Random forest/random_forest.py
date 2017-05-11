import numpy as np
from dtOmitted import DecisionNode
from dtOmitted import build_tree
from decision_tree import DecisionTree


class RandomForest(object):
    """
    RandomForest a class, that represents Random Forests.

    :param num_trees: Number of trees in the random forest
    :param max_tree_depth: maximum depth for each of the trees in the forest.
    :param ratio_per_tree: ratio of points to use to train each of
        the trees.
    """
    def __init__(self, num_trees, max_tree_depth, ratio_per_tree=0.5):
        self.num_trees = num_trees
        self.max_tree_depth = max_tree_depth
        self.trees = None
        self.ratio_per_tree=ratio_per_tree
        

    def fit(self, X, Y):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :param Y: 1 dimensional python list or numpy 1 dimensional array
        """
        n=len(X)
        idx = np.arange(n)
        np.random.seed(np.random.randint(1000))
        np.random.shuffle(idx)
        X = X[idx]
        Y = Y[idx]
        
        N=int(n*self.ratio_per_tree)
        
        self.trees = []

        for i in range(self.num_trees):
            idx=np.array([np.random.randint(n) for _ in range(N)])

            X1=X[idx]
            Y1=Y[idx]

            tr = DecisionTree(self.max_tree_depth)
            tr.fit(X1,Y1)
            self.trees.append(tr)


    def predict(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :return: (Y, conf), tuple with Y being 1 dimension python
        list with labels, and conf being 1 dimensional list with
        confidences for each of the labels.
        """
        l=np.array([self.trees[i].predict(X) for i in range(self.num_trees)]).transpose()
        Y=[]
        conf=[]

        for i in range(len(l)):
            if l[i].sum()/float(len(l[i])) > 0.5:
                Y.append(1.0)
                conf.append(l[i].sum()/float(len(l[i])))
            else:
                Y.append(0.0)
                conf.append(1-l[i].sum()/float(len(l[i])))

        return (Y, conf)
