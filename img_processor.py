import cv2 as cv
import numpy as np
import random
import os
from skimage.measure import block_reduce
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize


class ImgProcessor:
    def __init__(self, config):
        self.config = config['preprocess']
        self.pools = dict()
        self.pools['train'] = []
        self.pools['test'] = []

        self.process = self.config['process_list']

        self.resize = self.config['resize'][0]
        self.norm = self.config['norm']
        self.n_comps = self.config['sne_n_comps']
        self.filter_size = self.config['mp_filter_size']

        ok_count, ng_count = 0, 0

        for filename in os.listdir(self.config['data_path']+'OK/'):
            if ok_count < self.config['train_size']:
                self.pools['train'].append(self.config['data_path']+'OK/' + filename)
            else:
                self.pools['test'].append(self.config['data_path']+'OK/' + filename)
            ok_count += 1

        for filename in os.listdir(self.config['data_path']+'NG/'):
            if ng_count < self.config['train_size']:
                self.pools['train'].append(self.config['data_path']+'NG/' + filename)
            else:
                self.pools['test'].append(self.config['data_path']+'NG/' + filename)
            ng_count += 1

        print('%s imgs in train/' % str(len(self.pools['train'])))
        print('%s imgs in test/' % str(len(self.pools['test'])))

        random.shuffle(self.pools['train'])
        random.shuffle(self.pools['test'])

    def get_data(self, mode, flatten=True, save=False):
        """
        available process = resize, max_pooling, tsne

        output:
        imgs.shape = (data_size, feature_size)

        """
        imgs = []
        labels = []

        if mode == 'train':
            pool = self.pools['train']

        elif mode == 'test':
            pool = self.pools['test']

        for img_path in pool:

            img = cv.imread(img_path)
            img = self.__greyscale(img)
            if 'resize' in self.process:
                img = cv.resize(img, (self.resize, self.resize))
            if 'max_pooling' in self.process:
                img = self.__max_pooling(img)
            if flatten:
                img = img.flatten()
            imgs.append(img)

            if 'OK' in img_path:
                labels.append(0)
            elif 'NG' in img_path:
                labels.append(1)

        labels = np.asarray(labels)
        imgs = np.asarray(imgs)

        if self.norm:
            imgs = normalize(imgs)

        else:
            imgs = imgs/255

        if 'tsne' in self.process:
            imgs = TSNE(n_components=self.n_comps, perplexity=50, method='exact').fit_transform(imgs)

        if save:
            np.save(self.config['data_path'] + mode + '_X.npy', imgs)
            np.save(self.config['data_path'] + mode + '_Y.npy', labels)

        return imgs, labels

    def load_npy(self, mode):
        imgs = np.load(self.config['data_path'] + mode + '_X.npy')
        labels = np.load(self.config['data_path'] + mode + '_Y.npy')

        return imgs, labels

    def __greyscale(self, img):
        return cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    def __max_pooling(self, img):
        ax0, ax1 = self.filter_size[0], self.filter_size[1]
        return block_reduce(img, (ax0, ax1), func=np.max)
