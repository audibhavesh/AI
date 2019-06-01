import pickle as pc
import numpy as np
class Cifar_dataset:
    def __init__(self):
        pass

    def unpickle(self, filename):
        with open(filename, 'rb') as f:
            dict = pc.load(f, encoding='bytes')
        return dict

    def cifar10_data_unpickling(self, folder_name):
        cifar_meta_file = self.unpickle(folder_name + "/batches.meta")
        cifar_labels = [str(i, encoding='utf-8') for i in cifar_meta_file[b'label_names']]
        cifar_labels = np.array(cifar_labels)
        # print((cifar_labels))
        cifar_train_data = None
        cifar_train_labels = []
        for i in range(1, 6):
            cifar_train_set = self.unpickle(folder_name + "/data_batch_{}".format(i))
            if i == 1:
                cifar_train_data = cifar_train_set[b'data']
            else:
                cifar_train_data = np.vstack((cifar_train_data, cifar_train_set[b'data']))
            cifar_train_labels += cifar_train_set[b'labels']

        cifar_train_data = cifar_train_data.reshape(len(cifar_train_data), 3, 32, 32)
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
        cifar_train_labels = np.array(cifar_train_labels)

        cifar_test_set = self.unpickle(folder_name + "/test_batch")
        cifar_test_labels = cifar_test_set[b'labels']
        cifar_test_data = cifar_test_set[b'data']

        cifar_test_labels = np.array(cifar_test_labels)
        cifar_test_data = cifar_test_data.reshape(len(cifar_test_data), 3, 32, 32)
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)

        return cifar_train_data, cifar_train_labels, \
               cifar_test_data, cifar_test_labels
