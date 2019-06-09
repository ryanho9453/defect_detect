import sys
import json
import os

package_path = os.path.dirname(os.path.abspath(__file__)) + '/'
sys.path.append(package_path)

from evaluator import Evaluator
from model_picker import ModelPicker
from img_processor import ImgProcessor


with open(package_path + 'config.json') as f:
    config = json.load(f)


def main():
    img_pro = ImgProcessor(config)
    train_imgs, train_labels = img_pro.get_data(mode='train', save=True)
    test_imgs, true_y = img_pro.get_data(mode='test', save=True)

    # train_imgs, train_labels = img_pro.load_npy(mode='train')
    # test_imgs, true_y = img_pro.load_npy(mode='test')

    print('feature size : %s' % train_imgs.shape[1])

    model_picker = ModelPicker(config)
    evaluator = Evaluator(config)

    if 'svm' in model_picker.model_list:
        print('=======================')
        print('[SVM] training')
        model = model_picker.get_svm()
        model.fit(train_imgs, train_labels)
        pred_y = model.predict(test_imgs)
        print('[SVM] performance')
        evaluator.eval(true_y, pred_y)

    if 'rf' in model_picker.model_list:
        print('=======================')
        print('[Random Forest] training')
        model = model_picker.get_rf()
        model.fit(train_imgs, train_labels)
        pred_y = model.predict(test_imgs)
        print('[Random Forest] performance')
        evaluator.eval(true_y, pred_y)

    if 'src' in model_picker.model_list:
        print('=======================')
        print('[SRC] training')
        model = model_picker.get_src()
        pred_y = model.predict(train_imgs, train_labels, test_imgs)
        print('[SRC] performance')
        evaluator.eval(true_y, pred_y)


if __name__ == "__main__":
    main()
