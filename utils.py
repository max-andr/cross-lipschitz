"""
Various helping functions
"""
import pickle
import os
import numpy as np
import tensorflow as tf
import scipy.io


class Logger:
    def __init__(self, string):
        self.lst = [string]
        print(string)

    def add(self, string):
        self.lst.append(string)
        print(string)

    def clear(self):
        self.lst = []

    def get_str(self):
        return '\n'.join(self.lst)

    def to_file(self, folder, file_name):
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(folder + file_name, 'a') as f:
            f.write(self.get_str())


def sample_hyperparam(hyperparam_dict):
    while 1:
        hyperparam_set = []
        for param_name in hyperparam_dict:
            param_val = np.random.choice(hyperparam_dict[param_name])
            hyperparam_set.append(param_val)
        yield hyperparam_set


def save_var(var, f_name):
    with open(f_name, 'ab+') as file_write:
        pickle.dump(var, file_write)


def read_var(f_name):
    with open(f_name, 'rb') as file_read:
        return pickle.load(file_read)


def create_folders(folders):
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


def save_dicts_of_tf_vars(sess, model, exp_path, file_name, save_format):
    w, b = model.get_weight_dicts()
    param_names = ['w_' + n for n in w.keys()] + ['b_' + n for n in w.keys()]
    param_arrays = sess.run(list(w.values()) + list(b.values()))
    if save_format == 'txt':
        create_folders(['params' + exp_path + file_name])
        for param_name, param_array in zip(param_names, param_arrays):
            np.array('params{}{}/{}'.format(exp_path, file_name, param_name), param_array)
    elif save_format == 'mat':
        scipy.io.savemat('params{}{}.mat'.format(exp_path, file_name), dict(zip(param_names, param_arrays)))
    else:
        raise Exception('Wrong save_format')


def tf_print(tensor, msg=None, n=10):
    return tf.Print(tensor, [tensor], message=msg if msg else tensor.__repr__(), summarize=n)


def is_float(string):
    return string.replace('.', '').replace('-', '').replace('e', '').isdigit()
