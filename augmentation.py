import tensorflow as tf
import numpy as np
import regularizers


def random_crop_batch(batch_x, height, real_height, width, real_width):
    h_diff, w_diff = width - real_height, width - real_width
    batch_x_cropped_list = []
    for x in batch_x:
        h_rand, w_rand = int(h_diff * np.random.rand()), int(w_diff * np.random.rand())
        batch_x_cropped_list.append(x[h_rand:height - h_diff + h_rand, w_rand:width - w_diff + w_rand, :])
    return np.array(batch_x_cropped_list)


def augment_train(img_tensor, hps):
    def augment_each(img):
        # img = tf.random_crop(img, [hps.real_height, hps.real_width, hps.n_colors])
        if hps.dataset not in ['mnist', 'gtrsrb'] and hps.fl_mirroring:
            img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.minimum(tf.maximum(img, 0.0), 1.0)
        img = tf.image.random_contrast(img, lower=0.6, upper=1.4)
        img = tf.minimum(tf.maximum(img, 0.0), 1.0)
        return img

    with tf.device('/cpu:0'):
        if hps.fl_rotations:
            rand_angles = tf.random_uniform([hps.batch_size], minval=-hps.max_rotate_angle, maxval=hps.max_rotate_angle)
            img_tensor = tf.contrib.image.rotate(img_tensor, rand_angles)
        img_tensor = tf.map_fn(augment_each, img_tensor)

        if hps.gauss_noise_flag:
            expected_noise_norm = 2
            gauss_noise = tf.random_normal(tf.shape(img_tensor), stddev=expected_noise_norm / hps.n_input)
            img_tensor += gauss_noise
            img_tensor = tf.minimum(tf.maximum(img_tensor, 0.0), 1.0)
        return img_tensor


def augment_test(img_tensor, hps):
    def prepare_each(img):
        img = tf.image.central_crop(img, hps.real_height / hps.height)  # crop factor is 28/32
        return img

    with tf.device('/cpu:0'):
        img_tensor = tf.map_fn(prepare_each, img_tensor)
        return img_tensor


def adv_train(X_batch, Y_batch, model, hps, flag_train):
    n_adv = int(hps.alpha_adv * hps.batch_size)
    X_batch_adv, Y_batch_adv = X_batch[:n_adv], Y_batch[:n_adv]

    model.build_graph(X_batch_adv)
    f = model.get_logits()

    if 'weight_decay' in hps.reg_type:
        reg = regularizers.weight_decay(var_pattern='weights')
    elif 'cross_lipschitz' in hps.reg_type:
        if hps.nn_type == 'mlp1layer':
            reg = regularizers.cross_lipschitz_analytical_1hl(model, hps.n_ex, hps)
        else:
            reg = regularizers.cross_lipschitz(f, X_batch, hps.n_ex, hps)
    else:
        reg = tf.constant(0.0)  # 'dropout' and 'no' cases go here

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=f, labels=Y_batch_adv)) + hps.lmbd * reg

    eps = tf.random_uniform([n_adv], minval=hps.at_min_eps, maxval=hps.at_max_eps)
    # eps = tf.reshape(tf.tile(eps, [n_input_real]), [n_adv, n_input_real])
    grads = tf.gradients(loss, X_batch_adv)[0]
    if hps.at_method == 'fast_linf':
        eps = tf.transpose(tf.reshape(tf.tile(eps, [hps.n_input_real]), [hps.n_input_real, n_adv]))
        delta = eps * tf.sign(grads)
    elif hps.at_method == 'fast_l2':
        k = eps / tf.norm(grads, ord=2, axis=1)
        k = tf.transpose(tf.reshape(tf.tile(k, [hps.n_input_real]), [hps.n_input_real, n_adv]))
        delta = k * grads
    X_adv = tf.stop_gradient(tf.clip_by_value(X_batch_adv + delta, 0, 1))
    return tf.concat([X_adv, X_batch[n_adv:]], axis=0)
