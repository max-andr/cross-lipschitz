"""
The main module to train a neural network with specified settings, which should be called with command line arguments, e.g.:
python worker.py --experiment_name=advers_training --adv_train_flag --dataset=mnist --nn_type=mlp1layer --gpu_number=0
--gpu_memory=0.02 --reg_type=cross_lipschitz --lr=0.2 --lmbd=0.0001 --batch_size=64 --n_epochs=100
"""
import tensorflow as tf
import numpy as np
import os
import time
import utils
import data
import scipy.io
import glob
import models
import augmentation
import argparse
import regularizers
import ae_generation

np.set_printoptions(suppress=True, precision=4)


def f_activation(hps):
    if hps.activation_type == 'softplus':
        return lambda x: 1 / hps.softplus_alpha * tf.nn.softplus(hps.softplus_alpha * x)
    elif hps.activation_type == 'relu':
        return tf.nn.relu
    elif hps.activation_type == 'sigmoid':
        return tf.sigmoid
    else:
        raise Exception('Unknown activation function')


def eval_in_batches(X_tf, X_np, Y_tf, Y_np, sess, tensors):
    """Get all predictions for a dataset by running it in small batches."""
    n_batches = len(X_np) // hps.batch_size
    vals_total = [0] * len(tensors)
    for batch_x, batch_y in data.get_next_batch(X_np, Y_np, hps.batch_size):
        vals = sess.run(tensors, feed_dict={X_tf: batch_x, Y_tf: batch_y, flag_train: False, flag_take_grads: False})
        for i in range(len(vals)):
            vals_total[i] += vals[i]
    return [val_total / n_batches for val_total in vals_total]


def gradients_for_matlab(f, X_input, grad_class, n_input_real):
    # with tf.device('/cpu:0'):
    grad = tf.gradients(f[:, grad_class], X_input)[0]
    return tf.reshape(grad, [-1, n_input_real])


def create_hps_str(hps):
    # We can't take all hps for file names, so we select the most important ones
    hyperparam_str = "reg_type={} n_epochs={} lr={} lmbd={} keep_hidden={} batch_size={} augm_flag={} adv_train_flag={}". \
        format(hps.reg_type, hps.n_epochs, hps.lr, hps.lmbd, hps.keep_hidden, hps.batch_size, hps.augm_flag,
               hps.adv_train_flag)
    return hyperparam_str


def create_mixed_ae_batch(x_batch, x_tf, model, sess, hps):
    n_ae = round(hps.alpha_adv * len(x_batch))
    x_ae_batch, _ = ae_generation.generate_ae(x_batch[:n_ae], x_tf, model, sess, hps, min_over_all_classes=False,
                                              binary_search=False, batch_mode=False)
    return np.vstack([x_ae_batch, x_batch[n_ae:]])


parser = argparse.ArgumentParser(description='Define hyperparameters.')
parser.add_argument('--gpu_number', type=str, default='3', help='GPU number to use.')
parser.add_argument('--gpu_memory', type=float, default='0.05', help='GPU memory fraction to use.')
parser.add_argument('--experiment_name', type=str, default='test',
                    help='Name of the experiment, which is used to save the results/metrics/model in a certain folder.')
parser.add_argument('--nn_type', type=str, default='mlp1layer',
                    help='NN type: resnet, mlp1layer, cnn_basic, cnn_advanced. Only the first 2 were reported in '
                         'the paper.')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='mnist, cifar10, gtrsrb (German traffic roadsign dataset).')
parser.add_argument('--reg_type', type=str, default='cross_lipschitz',
                    help='cross_lipschitz, no, weight_decay, dropout')
parser.add_argument('--opt_method', type=str, default='sgd',
                    help='Optimization method: sgd or momentum (default momentum: 0.9)')
parser.add_argument('--n_epochs', type=int, default=200, help='Number of epochs.')
parser.add_argument('--lr', type=float, default=0.2, help='Learning rate.')
parser.add_argument('--lmbd', type=float, default=0.1, help='Regularization parameter for WD or CL.')
parser.add_argument('--keep_hidden', type=float, default=0.5, help='Probability of keeping a neuron for dropout')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size, which is also used for evaluating '
                                                                'train/validation/test error by parts.')
parser.add_argument('--augm_flag', action='store_true',
                    help='Data augmentation: rotation, mirroring (not for mnist and gtrsrb), gauss noise.')
parser.add_argument('--adv_train_flag', action='store_true',
                    help='Adversarial training (data augmentation with adversarial examples)')
parser.add_argument('--continue_train', action='store_true', help='True if we want to continue training for more epochs.')
parser.add_argument('--eval_robustness', action='store_true',
                    help='True if we want to calculate the robustness upper bounds with the proposed method to '
                         'generate adversarial examples wrt L2 norm. We reuse the same computational graph, '
                         'so it is also implemented in this module.')

hps = parser.parse_args()  # returns a Namespace object, new fields can be set like hps.abc = 10

hps_str = create_hps_str(hps)
log = utils.Logger('The script started on GPU {} with hyperparameters: {}'.format(hps.gpu_number, hps_str))

hps.lr_decay_scheme = 'manual'  # 'manual' or 'exponential'
hps.r_seed = 5  # random seed for initialization
if hps.nn_type == 'mlp1layer' or 'cnn' in hps.nn_type:
    hps.keep_input = 0.9
if hps.nn_type == 'mlp1layer':
    hps.n_hidden1 = 1024  # only for MLP
    hps.bias_flag = False
if 'cnn' in hps.nn_type:
    hps.keep_conv = 1.0  # 0.75 will give better results, but train for more time.
    hps.scale_init = 0.1  # scaling factor for random init in CNN. default=1
if hps.nn_type == 'resnet':
    hps.n_resid_units = 5  # regulates the depth of ResNet
    hps.use_bottleneck = False  # bottleneck in ResNet
hps.activation_type = 'softplus'  # 'softplus' or 'relu'
if hps.activation_type == 'softplus':
    hps.softplus_alpha = 10
hps.merge_valid = True
hps.fl_rotations, hps.fl_mirroring, hps.max_rotate_angle, hps.gauss_noise_flag = False, True, np.pi / 20, False
if hps.adv_train_flag:
    # either 'targeted_box_constrained' (the proposed method), 'fast_linf' or 'fast_l2'
    hps.at_method = 'targeted_box_constrained'
    hps.alpha_adv = 0.5
    hps.at_min_eps, hps.at_max_eps = (0.05, 0.15) if hps.at_method == 'fast_linf' else (0.2, 2.0)
hps.eval_rob_each_n = 15  # each hps.eval_rob_each_n epoch we evaluate the robustness and store it with other metrics
hps.n_rob_each_epoch = 100  # number of test examples to evaluate the robustness on each hps.eval_rob_each_n epoch
hps.as_image = False if 'mlp' in hps.nn_type else True
hps.save_f_and_fgrad = False  # should we import f_i and f_grad_i as .mat files
hps.activation = f_activation(hps)

X_train, Y_train, X_validation, Y_validation, X_test, Y_test = data.get_dataset(hps.dataset, as_image=True)
if hps.merge_valid:
    X_train = np.concatenate([X_train, X_validation])
    Y_train = np.concatenate([Y_train, Y_validation])

hps.n_ex_train, hps.height, hps.width, hps.n_colors = X_train.shape
if hps.augm_flag:
    hps.real_height, hps.real_width = (28, 28) if hps.dataset in ['cifar10', 'gtrsrb'] else (24, 24)
    margin = (hps.height - hps.real_height) // 2
    # For data augmentation we need X_train to be centrally cropped to evaluate the performance on training set
    X_train_eval = X_train[:, margin:hps.height - margin, margin:hps.width - margin, :]
    X_validation = X_validation[:, margin:hps.height - margin, margin:hps.width - margin, :]
    X_test = X_test[:, margin:hps.height - margin, margin:hps.width - margin, :]
else:
    hps.real_height, hps.real_width = hps.height, hps.width
hps.n_ex_valid = len(X_validation)
hps.n_classes = Y_train.shape[1]
hps.n_input_real = hps.real_height * hps.real_width * hps.n_colors
if hps.lr_decay_scheme == 'manual':
    hps.n_updates_total = hps.n_epochs * hps.n_ex_train // hps.batch_size
    # original ResNet paper: [32000, 48000] or [0.72, 0.88] with n_epochs=160
    hps.lr_decay_n_updates = [round(0.72 * hps.n_updates_total), round(0.88 * hps.n_updates_total)]
elif hps.lr_decay_scheme == 'exponential':
    hps.lr_decay = 0.985
# All available Neural Network models
models_dict = {'mlp1layer':    models.MLP1Layer,
               'cnn_basic':    models.CNNBasic,
               'cnn_advanced': models.CNNAdvanced,
               'resnet':       models.ResNet
               }

# Define the computational graph
device = '/gpu:0'
with tf.device(device):
    tf.reset_default_graph()
    extra_train_ops = []
    tf_n_updates = tf.Variable(0, trainable=False)

    flag_train = tf.placeholder(tf.bool, name='is_training')
    flag_take_grads = tf.placeholder(tf.bool, name='is_training')
    grad_class = tf.placeholder(tf.int32)
    X = tf.placeholder("float", [None, hps.real_height, hps.real_width, hps.n_colors])
    Y = tf.placeholder("float", [None, hps.n_classes])
    X_input = tf.identity(X)
    hps.n_ex = tf.shape(X)[0]

    model = models_dict[hps.nn_type](flag_train, flag_take_grads, hps)

    if hps.augm_flag:  # Data augmentation except random cropping is implemented inside the TF comp. graph
        X_input = tf.cond(flag_train, lambda: augmentation.augment_train(X_input, hps), lambda: X_input)
    if not hps.as_image:  # If we treat the image like a long vector without spatial structure
        X_input = tf.reshape(X_input, [-1, hps.n_input_real])
    # We implemented fast L_inf or L_2 methods inside the computational graph
    if hps.adv_train_flag and hps.at_method != 'targeted_box_constrained':
        X_input = tf.cond(flag_train, lambda: augmentation.adv_train(X_input, Y, model, hps, flag_train), lambda: X_input)

    model.build_graph(X_input)
    f = model.get_logits()
    extra_train_ops += model.extra_train_ops

    grad_tensor = gradients_for_matlab(f, X_input, grad_class, hps.n_input_real)

    if 'weight_decay' in hps.reg_type:
        reg = regularizers.weight_decay(var_pattern='weights')
    elif 'cross_lipschitz' in hps.reg_type:
        if hps.nn_type == 'mlp1layer':
            reg = regularizers.cross_lipschitz_analytical_1hl(model, hps.n_ex, hps)
        else:
            reg = regularizers.cross_lipschitz(f, X_input, hps.n_ex, hps)
    else:
        reg = tf.constant(0.0)  # 'dropout' and 'no' cases go here

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=f, labels=Y)) + hps.lmbd * reg

    if hps.lr_decay_scheme == 'exponential':
        lr_tf = tf.train.exponential_decay(hps.lr, tf_n_updates * hps.batch_size, hps.n_ex_train, hps.lr_decay,
                                           staircase=True)
    elif hps.lr_decay_scheme == 'manual':
        lr_tf = tf.train.piecewise_constant(tf_n_updates, hps.lr_decay_n_updates, [hps.lr, hps.lr * 0.1, hps.lr * 0.01])
    else:
        raise Exception('Wrong lr decay scheme.')

    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(loss, trainable_variables)
    if hps.opt_method == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr_tf)
    elif hps.opt_method == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr_tf, momentum=0.9)
    else:
        raise Exception('Wrong optimizer.')

    apply_op = optimizer.apply_gradients(zip(grads, trainable_variables), global_step=tf_n_updates, name='train_step')

    train_ops = [apply_op] + extra_train_ops
    train = tf.group(*train_ops)

    y_pred, y_true = tf.argmax(f, 1), tf.argmax(Y, 1)
    errors_boolean = tf.cast(tf.not_equal(y_pred, y_true), "float")
    error_rate = tf.reduce_mean(errors_boolean)

    saver = tf.train.Saver()

    gpu_options = tf.GPUOptions(visible_device_list=hps.gpu_number,
                                per_process_gpu_memory_fraction=hps.gpu_memory)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

with tf.Session(config=config) as sess:
    # We need to restore the trained model if we want to continue training or to evaluate the robustness
    if hps.continue_train or hps.eval_robustness:
        f_model_pattern = 'models/{}/{}/{}/*{}*'.format(hps.experiment_name, hps.dataset, hps.reg_type, hps_str)
        print(f_model_pattern)
        f_model = glob.glob(f_model_pattern)[0]
        f_model = '.'.join(f_model.split('.')[:-1])  # take name of file without .index or other stuff
        print('Restored model:', f_model)
        saver.restore(sess, f_model)

        metrics_file = 'metrics/' + '/'.join(f_model.split('/')[1:])
        metrics_np = np.loadtxt(metrics_file)
        metrics = [metrics_np.tolist()] if len(metrics_np.shape) == 1 else metrics_np.tolist()
    else:
        sess.run(tf.global_variables_initializer())  # run 'init' op
        metrics = []
    log.add('Session started with hyperparameters: {} \n'.format(hps_str))
    time_start = time.time()
    exp_path = '/{}/{}/{}/'.format(hps.experiment_name, hps.dataset, hps.reg_type)

    if not hps.eval_robustness:
        for epoch in range(1, hps.n_epochs + 1):
            # Training
            for batch_x, batch_y in data.get_next_batch(X_train, Y_train, hps.batch_size, hps.augm_flag):
                if hps.augm_flag:  # random cropping for the data augmentation
                    batch_x = augmentation.random_crop_batch(batch_x, hps.height, hps.real_height, hps.width, hps.real_width)
                if hps.adv_train_flag and hps.at_method == 'targeted_box_constrained':
                    # The proposed method runs using numpy, not in tensorflow, so it should be done in each batch outside
                    # of the TF computational graph
                    batch_x = create_mixed_ae_batch(batch_x, X, model, sess, hps)
                sess.run([train], feed_dict={X: batch_x, Y: batch_y, flag_train: True, flag_take_grads: False})
            # Evaluating different metrics once per epoch
            loss_val, reg_val, error_rate_train = eval_in_batches(X, X_train_eval if hps.augm_flag else X_train, Y,
                                                                  Y_train, sess, [loss, reg, error_rate])
            error_rate_valid, = eval_in_batches(X, X_validation, Y, Y_validation, sess, [error_rate])
            error_rate_test, _ = eval_in_batches(X, X_test, Y, Y_test, sess, [error_rate, loss])
            if epoch % hps.eval_rob_each_n == 1:  # Evaluate the robustness each n epochs
                random_idx = np.random.permutation(len(X_test))[:hps.n_rob_each_epoch]
                _, min_norms = ae_generation.generate_ae(X_test[random_idx], X, model, sess, hps, min_over_all_classes=True,
                                                         binary_search=True, batch_mode=True)
                avg_delta_norm = np.mean(min_norms[min_norms != np.inf])
            metrics.append([loss_val, error_rate_train, error_rate_valid, error_rate_test, avg_delta_norm])
            log.add(
                'Epoch: {:d} valid err: {:.3f}% test err: {:.3f}% train err: {:.3f}% avg_d_norm: {:.3f} loss: {:.5f} '
                'reg: {:.5f}'.format(epoch, error_rate_valid * 100, error_rate_test * 100,
                                     error_rate_train * 100, avg_delta_norm, loss_val - hps.lmbd * reg_val,
                                     hps.lmbd * reg_val))

        log.add('Finished this set of hyperparameters in {:.2f} min'.format((time.time() - time_start) / 60))
        file_name = 'valid={:.2f} test={:.2f} {} '.format(error_rate_valid * 100, error_rate_test * 100, hps_str)

        utils.create_folders(
                [base_folder + exp_path for base_folder in ['logs', 'models', 'metrics', 'params', 'diffs']])
        log.to_file('logs' + exp_path, file_name)  # save optimization output
        np.savetxt('metrics' + exp_path + file_name, np.array(metrics))  # save optimization metrics for future plots
        saver.save(sess, 'models' + exp_path + file_name)  # save TF model for future real robustness test
        if 'mlp' in hps.nn_type:
            # (obsolete) We save all parameters to use them in Matlab
            utils.save_dicts_of_tf_vars(sess, model, exp_path, file_name, save_format='mat')
    else:
        x_ae, min_norms = ae_generation.generate_ae(X_test, X, model, sess, hps, min_over_all_classes=True,
                                                    binary_search=True, batch_mode=True, verbose=False)
        file_name = f_model.split('/')[-1]
        utils.create_folders(['deltas' + exp_path])
        np.savetxt('deltas' + exp_path + file_name, min_norms)
        if hps.save_f_and_fgrad:  # import f_i and f_grad_i as .mat files
            # (obsolete) calculate f and nabla f wrt x for the robustness test initially implemented in Matlab
            grads_val = np.zeros([len(X_test), hps.n_input_real, hps.n_classes])
            for cl in range(hps.n_classes):
                grads_val[:, :, cl] = sess.run(grad_tensor, feed_dict={X:          X_test, Y: Y_test, flag_train: False,
                                                                       grad_class: cl, flag_take_grads: True})
            f_val = sess.run([f], feed_dict={X: X_test, Y: Y_test, flag_train: False, flag_take_grads: False})

            param_names, param_arrays = ['f_val', 'grad_val'], [f_val, grads_val]
            scipy.io.savemat('diffs{}{}.mat'.format(exp_path, file_name), dict(zip(param_names, param_arrays)))
log.add('Worker done: {} in {:.2f} min\n\n'.format(hps_str, (time.time() - time_start) / 60))
