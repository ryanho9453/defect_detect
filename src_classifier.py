import numpy as np
import os
from sklearn.decomposition import SparseCoder
from sklearn.preprocessing import normalize


package_path = os.path.dirname(os.path.abspath(__file__)) + '/'


class SrcClassifier:
    def __init__(self, config):
        self.config = config['src']
        self.coder_alpha = self.config['alpha']

    def predict(self, train_X, train_Y, test_X):
        """
        input : X.shape = (data_size, feature_size)

        """
        X = normalize(test_X)
        D = self.__gen_dict(train_X)
        Z = self.__sparse_encode(D, test_X)

        test_size = Z.shape[0]

        pred_y = []
        for i in range(test_size):
            Zi = Z[i, :][:, np.newaxis]
            pred = Zi * D
            Xi = X[i, :]
            Ei = np.power(np.sum(pred - Xi, axis=1), 2)
            train_data_idx = int(np.argmin(Ei))
            Ci = train_Y[train_data_idx]
            pred_y.append(Ci)

        return np.array(pred_y)

    def __gen_dict(self, train_X):
        """

        D.shape = (dict_size, feature_size)

        """

        D = normalize(train_X)

        return D

    def __sparse_encode(self, D, test_X):
        """

        Z.shape = (test_size, atoms_size=dict_size)

        """

        coder = SparseCoder(dictionary=D, transform_algorithm='lasso_cd', transform_alpha=self.coder_alpha)
        coder.fit(test_X)
        Z = coder.transform(test_X)

        return Z

