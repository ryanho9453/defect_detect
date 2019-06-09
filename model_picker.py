import os
import sys
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

package_path = os.path.dirname(os.path.abspath(__file__)) + '/'
sys.path.append(package_path)

from src_classifier import SrcClassifier


class ModelPicker:
    def __init__(self, config):
        self.config = config['model']
        self.model_list = self.config['model_list']

    def get_svm(self):
        c = self.config['svm']['c']
        gamma = self.config['svm']['gamma']
        clf = svm.SVC(kernel='rbf', C=c, gamma=gamma)

        return clf

    def get_rf(self):
        n_est = self.config['rf']['n_estimators']
        max_dep = self.config['rf']['max_depth']

        clf = RandomForestClassifier(n_estimators=n_est, max_depth=max_dep, random_state=0)

        return clf

    def get_src(self):
        src = SrcClassifier(self.config)

        return src
