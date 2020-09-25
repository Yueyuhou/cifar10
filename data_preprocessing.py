import numpy as np
import os
import glob
import tensorflow as tf
from tensorflow import keras


class DataLoading():
    """
    load_data:
    return train/test data and labels(after one-hot)

    """
    def __init__(self, data_path, num_class):
        self.num_class = num_class
        self.data_path = data_path

    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def load_data(self):
        train_data_path = glob.glob(os.path.join(self.data_path, 'data*'))
        test_data_path = glob.glob(os.path.join(self.data_path, 'test*'))
        train_data_dict = [self.unpickle(path) for path in train_data_path]
        test_data_dict = [self.unpickle(path) for path in test_data_path]

        train_data_img = np.asarray([dic[b'data'] for dic in train_data_dict])
        train_data_img = train_data_img.reshape(-1, 3, 32, 32)
        train_data_label = np.asarray([dic[b'labels'] for dic in train_data_dict])
        train_data_label = train_data_label.reshape(-1, 1)
        train_data_label = keras.utils.to_categorical(train_data_label, self.num_class)


        test_data_img = np.asarray([dic[b'data'] for dic in test_data_dict])
        test_data_img = test_data_img.reshape(-1, 3, 32, 32)
        test_data_label = np.asarray([dic[b'labels'] for dic in test_data_dict])
        test_data_label = test_data_label.reshape(-1, 1)
        test_data_label = keras.utils.to_categorical(test_data_label, self.num_class)

        return train_data_img, train_data_label, test_data_img, test_data_label












