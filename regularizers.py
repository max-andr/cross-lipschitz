import tensorflow as tf
import numpy as np


def cross_lipschitz(f, x, n_ex, hps):
    """
    Calculates Cross-Lipschitz regularization in a straightforward way using tf.gradients to calculate the required
    derivatives. Suitable for all differentiable classifiers. It is calculated for the given batch (details:
    "Formal Guarantees on the Robustness of a Classifier against Adversarial Manipulation",
    http://www.ml.uni-saarland.de/Publications/HeiAnd-FormGuarAdvManipLongVersion.pdf).

    f: tensor, (batch_size, n_classes) - values of the output layer before softmax for a batch
    x: tensor, (batch_size, n_input) or (batch_size, height, width, color) - input images
    n_classes: int - number of classes
    n_ex: int - number of examples in a batch
    """
    n_summations = tf.cast(hps.n_classes ** 2 * n_ex, tf.float32)  # normalizing factor to unify scale of lambda across datasets
    reg = 0
    grad_matrix_list = [tf.gradients(f[:, k], x)[0] for k in range(hps.n_classes)]  # take each gradient wrt input only once
    if hps.as_image:  # if x has shape (batch_size, height, width, color), then we need to flatten it first
        grad_matrix_list = [tf.reshape(grad, [-1, hps.real_height * hps.real_width * hps.n_colors]) for grad in grad_matrix_list]
    for l in range(hps.n_classes):
        for m in range(l + 1, hps.n_classes):
            grad_diff_matrix = grad_matrix_list[l] - grad_matrix_list[m]  # difference of gradients for a class pair (l, m)
            norm_for_batch = tf.norm(grad_diff_matrix, ord=2, axis=1)
            reg += 2 * tf.reduce_sum(tf.square(norm_for_batch))  # 2 comes from the fact, that we do summation only for distinct pairs (l, m)
    return reg / n_summations

def cross_lipschitz_updated(f, x, y, n_ex, hps):
    """
    Calculates Cross-Lipschitz regularization in a straightforward way using tf.gradients to calculate the required
    derivatives. Suitable for all differentiable classifiers. It is calculated for the given batch (details:
    "Formal Guarantees on the Robustness of a Classifier against Adversarial Manipulation",
    http://www.ml.uni-saarland.de/Publications/HeiAnd-FormGuarAdvManipLongVersion.pdf).

    f: tensor, (batch_size, n_classes) - values of the output layer before softmax for a batch
    x: tensor, (batch_size, n_input) or (batch_size, height, width, color) - input images
    y: tensor, (batch_size) - input image labels
    n_classes: int - number of classes
    n_ex: int - number of examples in a batch
    """
    n_summations = tf.cast(hps.n_classes ** 2 * n_ex, tf.float32)  # normalizing factor to unify scale of lambda across datasets
    reg = 0
    grad_matrix_list = [tf.gradients(f[:, k], x)[0] for k in range(hps.n_classes)]  # take each gradient wrt input only once
    if hps.as_image:  # if x has shape (batch_size, height, width, color), then we need to flatten it first
        grad_matrix_list = [tf.reshape(grad, [-1, hps.real_height * hps.real_width * hps.n_colors]) for grad in grad_matrix_list]

    for inputIter in range(len(y)):
        maxClsGradIdx = 0
        for m in range(1, hps.n_classes):
            if tf.norm(grad_matrix_list[m], ord=2, axis=1) > tf.norm(grad_matrix_list[maxClsGradIdx], ord=2, axis=1):
                maxClsGradIdx = m

        grad_diff_matrix = grad_matrix_list[l] - grad_matrix_list[maxClsGradIdx]  # difference of gradients for a class pair (l, m)
        norm_for_batch = tf.norm(grad_diff_matrix, ord=2, axis=1)

        grad_diff_matrix = grad_matrix_list[l] - grad_matrix_list[maxClsGradIdx]  # difference of gradients for a class pair (l, m)
        norm_for_batch = tf.norm(grad_diff_matrix, ord=2, axis=1)

        reg += 2 * tf.reduce_sum(tf.square(norm_for_batch))  # 2 comes from the fact, that we do summation only for distinct pairs (l, m)
    return reg / n_summations


def cross_lipschitz_analytical_1hl(model, n_ex, hps):
    """
    Calculates Cross-Lipschitz regularization in analytic form for 1 hidden layer Neural Network. (details:
    "Formal Guarantees on the Robustness of a Classifier against Adversarial Manipulation",
    http://www.ml.uni-saarland.de/Publications/HeiAnd-FormGuarAdvManipLongVersion.pdf).

    model: MLP1Layer object (see models.py)
    n_ex: int - number of examples in a batch
    hps: an object with fields n_classes (number of classes) and activation_type (type of activation function).
    """
    def der_activation(hidden1):
        if hps.activation_type == 'softplus':
            return tf.nn.sigmoid(hps.softplus_alpha * hidden1)
        elif hps.activation_type == 'relu':
            return tf.cast(hidden1 > 0, tf.float32)
        else:
            raise Exception('Unknown activation function')

    w, hidden1 = model.get_weight_dicts()[0], model.get_hidden1()

    n_summations = tf.cast(hps.n_classes ** 2 * n_ex, tf.float32)
    sigm_der = der_activation(hidden1)
    sigm_der_outprod = tf.matmul(tf.transpose(sigm_der), sigm_der)
    u_outprod = tf.matmul(tf.transpose(w['h1']), w['h1'])
    w_outprod = tf.matmul(w['out'], tf.transpose(w['out']))
    sumw = tf.reduce_sum(w['out'], axis=1, keep_dims=True)
    sumw_outprod = tf.matmul(sumw, tf.transpose(sumw))
    reg_matrix = (hps.n_classes * w_outprod - sumw_outprod) * sigm_der_outprod * u_outprod
    reg = 2 * tf.reduce_sum(reg_matrix)
    return reg / n_summations


def weight_decay(var_pattern):
    """
    L2 weight decay loss, based on all weights that have var_pattern in their name

    var_pattern - a substring of a name of weights variables that we want to use in Weight Decay.
    """
    costs = []
    for var in tf.trainable_variables():
        if var.op.name.find(var_pattern) != -1:
            costs.append(tf.nn.l2_loss(var))
    return tf.add_n(costs)
