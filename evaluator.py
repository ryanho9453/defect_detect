from sklearn.metrics import f1_score, confusion_matrix


class Evaluator:
    def __init__(self, config):
        self.config = config

    def eval(self, true_y, pred_y):
        score = f1_score(y_true=true_y, y_pred=pred_y, average='weighted')
        matrix = confusion_matrix(y_true=true_y, y_pred=pred_y)
        print(score)
        print(matrix)
