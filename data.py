import numpy
import numpy as np
import csv
import scipy.io
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

import os
import sys
import wget
import tarfile
import shutil
if sys.version_info[0] == 3:
    import pickle
else:
    import cPickle as pickle

def get_next_batch(X, Y, batch_size, augm_flag=False):
    n_batches = len(X) // batch_size
    height, width = X.shape[1], X.shape[2]
    rand_idx = np.random.permutation(len(X))[:n_batches * batch_size]
    for batch_idx in rand_idx.reshape([n_batches, batch_size]):
        batch_x, batch_y = X[batch_idx], Y[batch_idx]
        yield batch_x, batch_y


def get_dataset(dataset, as_image):
    if dataset == 'mnist':
        return get_mnist(as_image)
    elif dataset == 'cifar10':
        return get_cifar10(as_image)
    elif dataset == 'gtrsrb':
        return get_gtrsrb(as_image)
    else:
        raise Exception("Wrong dataset name.")


def get_mnist(as_image=True, onehot=True):
    def reshape_mnist(x):
        x = np.reshape(x, [-1, 28, 28, 1])
        return x

    mnist = input_data.read_data_sets("data/mnist", one_hot=True, validation_size=10000)
    X_validation, Y_validation = mnist.validation.images, mnist.validation.labels
    X_train, Y_train = mnist.train.images, mnist.train.labels
    X_test, Y_test = mnist.test.images, mnist.test.labels
    if as_image:
        X_train = reshape_mnist(X_train)
        X_validation = reshape_mnist(X_validation)
        X_test = reshape_mnist(X_test)
    if not onehot:
        Y_test = np.argmax(Y_test, 1)
    return X_train, Y_train, X_validation, Y_validation, X_test, Y_test


def dense_to_one_hot(labels_dense):
    """Convert class labels from scalars to one-hot vectors."""
    num_classes = len(np.unique(labels_dense))
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def normalize(x_train, x_test):
    # return (x - x.min()) / (x.max() - x.min())
    # return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)#, 1.0/np.sqrt(x.shape[1]))
    # min and max are per-image
    # return (x - x.min(axis=1, keepdims=True)) / (x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True))

    x_train, x_test = x_train / 255.0, x_test / 255.0
    # x_train_pixel_mean = x_train.mean(axis=0)  # per-pixel mean
    # x_train = x_train - x_train_pixel_mean
    # x_test = x_test - x_train_pixel_mean
    return x_train, x_test


def get_cifar10(as_image=True, onehot=True, validation_size=5000):
    """load all CIFAR-10 data and merge training batches"""

    def download_cifar10_dataset(folder):
        archieveFileName = 'cifar-10-python.tar.gz'
        if os.path.exists(archieveFileName):
            os.remove(archieveFileName)
        print ("Downloading CIFAR-10 dataset")
        url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        filename = wget.download(url)

        # Extract the tar file
        print ("Extracting archive")
        tar = tarfile.open(filename)
        tar.extractall()
        tar.close()

        # Remove the data to this location
        shutil.move('cifar-10-batches-py', folder)
        os.remove(filename)

    def load_cifar10_file(filename):
        """load data from single CIFAR-10 file"""
        with open(filename, 'rb') as f:
            # data_dict = pickle.load(f, encoding='latin1')
            if sys.version_info[0] == 3:
                data_dict = pickle.load(f, encoding='latin1')
            else:
                data_dict = pickle.load(f)
            x = data_dict['data']
            y = data_dict['labels']
            x = x.astype(float)
            y = np.array(y)
            return x, y

    def reshape_cifar(x):
        x = np.reshape(x, (-1, 3, 32, 32))
        x = np.transpose(x, (0, 2, 3, 1))
        return x

    folder = 'data/cifar10/'
    if not os.path.exists(folder):
        download_cifar10_dataset(folder)

    xs, ys = [], []
    for i in range(1, 6):
        filename = folder + 'data_batch_' + str(i)
        X, Y = load_cifar10_file(filename)
        xs.append(X)
        ys.append(Y)

    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    del xs, ys

    x_test, y_test = load_cifar10_file(folder + 'test_batch')

    # Subtract mean image
    # mean_image = np.mean(x_train, axis=0)
    # x_train, x_test = x_train - mean_image, x_test - mean_image

    # Normalize all pixels to [0..1]
    x_train, x_test = normalize(x_train, x_test)

    # zca = zca_whitening_matrix(x_train)
    # x_train = np.dot(zca, x_train)
    # x_test = np.dot(zca, x_test)

    # Reshape data to 3 channels per pixel
    x_train, x_test = reshape_cifar(x_train), reshape_cifar(x_test)
    if not as_image:
        x_train, x_test = np.reshape(x_train, (-1, 32*32*3)), np.reshape(x_test, (-1, 32*32*3))
    if onehot:
        y_train, y_test = dense_to_one_hot(y_train), dense_to_one_hot(y_test)
    valid_idx = np.random.choice(len(x_train), size=validation_size, replace=False)
    X_validation, Y_validation = x_train[valid_idx, :], y_train[valid_idx]
    X_train, Y_train = np.delete(x_train, valid_idx, axis=0), np.delete(y_train, valid_idx, axis=0)
    X_test, Y_test = x_test, y_test
    return X_train, Y_train, X_validation, Y_validation, X_test, Y_test


def get_gtrsrb(as_image=True, onehot=True, validation_size=5000):
    def read_images(rootpath, data_part):
        """
        Reads traffic sign data for German Traffic Sign Recognition Benchmark.

        Arguments: path to the traffic sign data, for example './GTSRB/Training'
        Returns:   list of images, list of corresponding labels
        """
        h, w = 32, 32
        n_folders = 43 if data_part == 'train' else 1
        images = []  # images
        labels = []  # corresponding labels
        # loop over all 42 classes
        for cls in range(0, n_folders):
            if data_part == 'train':
                prefix = rootpath + format(cls, '05d') + '/'  # subdirectory for class
                f_annotation = prefix + 'GT-' + format(cls, '05d') + '.csv'
            else:
                prefix = rootpath
                f_annotation = prefix + 'GT-final_test.csv'
            gtFile = open(f_annotation)  # annotations file
            gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations
            # loop over all images in current annotations file
            for row in list(gtReader)[1:]:
                img = Image.open(prefix + row[0])
                img = img.resize((h, w), Image.ANTIALIAS)
                images.append(np.array(img))  # the 1th column is the filename
                labels.append(int(row[7]))  # the 8th column is the label
            gtFile.close()
        return np.array(images, dtype=np.uint8), np.array(labels)

    X_train, Y_train = read_images('data/gtrsrb/train/Final_Training/Images/', 'train')
    X_test, Y_test = read_images('data/gtrsrb/test/Images/', 'test')
    # print(X_train.shape, X_test.shape)
    X_train, X_test = normalize(X_train, X_test)
    if onehot:
        Y_train, Y_test = dense_to_one_hot(Y_train), dense_to_one_hot(Y_test)

    valid_idx = np.random.choice(len(X_train), size=validation_size, replace=False)
    X_validation, Y_validation = X_train[valid_idx, :], Y_train[valid_idx]
    X_train, Y_train = np.delete(X_train, valid_idx, axis=0), np.delete(Y_train, valid_idx, axis=0)
    return X_train, Y_train, X_validation, Y_validation, X_test, Y_test

if __name__ == '__main__':
    X_train, Y_train, X_validation, Y_validation, X_test, Y_test = get_gtrsrb(as_image=True, onehot=False)
    scipy.io.savemat('data/gtrsrb/gtrsrb.mat', {'Xtrain': X_train, 'Xtest': X_test,
                                                'Ytrain': Y_train, 'Ytest': Y_test})
