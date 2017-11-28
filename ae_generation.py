import numpy as np
import tensorflow as tf
import attacks
import time


def ae_binary_search(model, sess, x_tf, x_point, v, c, dataset, pred_class, n_classes, ae_gen_function, verbose=False):
    other_classes = np.array([cls for cls in range(n_classes) if cls != pred_class])
    eps = 0.01
    mult_coef_dict = {'mnist': 1.2, 'cifar10': 1.3, 'gtrsrb': 1.2}
    mult_coef = mult_coef_dict[dataset]

    # If CL or other very robust method - make first part as fast as possible
    if np.mean(np.abs(v)) < 0.01:
        mult_coef = 1.7

    cnt, last_succ_c, last_fail_c = 0, 0, 0
    # iterate until we will have 2 values of c that lead to AE on different sides of the decision boundary
    while last_succ_c == 0 or last_fail_c == 0:
        delta, opt_failed = ae_gen_function(x_point, v, c)
        norm_ae = np.linalg.norm(delta, 2)

        if opt_failed and cnt >= 10:
            print('failed to generate AE')
            return delta, True
        # if norm_ae < 0.000000001:
        #     print('AE is very close to dec. boundary with norm=', norm_ae)
        #     return delta, True

        if opt_failed:
            c /= mult_coef
            fl_real_ae = False
        else:
            # test if point is classified differently
            x_ae = x_point + delta
            f_point = model.run_logits(x_ae[np.newaxis], x_tf, sess)[0]
            if verbose:
                print('f init pred: {}, f top-2{}'.format(f_point[pred_class], np.sort(f_point)[-2:]))
            if np.max(f_point[other_classes]) > f_point[pred_class]:
                last_succ_delta, last_succ_c = delta, c
                c /= mult_coef
                fl_real_ae = True
            else:
                last_fail_c = c
                c *= mult_coef
                fl_real_ae = False
        if verbose:
            print('Success: {} {}  c={:.5f}  <v, d>={:.3f}  norm={:.5f}'.format(fl_real_ae, not opt_failed, c, v.dot(delta), norm_ae))
        cnt += 1
    if verbose:
        print('\n\n')

    # but in general we need a loop
    prev_ae_norm = 0
    cur_ae_norm = np.linalg.norm(delta, 2)
    while abs(cur_ae_norm - prev_ae_norm) > eps:
        c = (last_succ_c + last_fail_c) / 2
        delta, opt_failed = ae_gen_function(x_point, v, c)
        norm_ae = np.linalg.norm(delta, 2)

        prev_ae_norm, cur_ae_norm = cur_ae_norm, np.linalg.norm(delta, 2)
        # test if point is classified diffrently
        x_ae = x_point + delta
        f_point = model.run_logits(x_ae[np.newaxis], x_tf, sess)[0]
        if verbose:
            print('f init pred: {}, f top-2{}'.format(f_point[pred_class], np.sort(f_point)[-2:]))
        if np.max(f_point[other_classes]) > f_point[pred_class]:
            last_succ_delta, last_succ_c = delta, c
            fl_real_ae = True
        else:
            last_fail_c = c
            fl_real_ae = False
        if verbose:
            print('Success: {} {}  c={:.3f}  <v, d>={:.3f}  norm={:.5f}'.format(fl_real_ae, not opt_failed, c, v.dot(delta), norm_ae))
        cnt += 1
    if verbose:
        print('\n\n')
    return last_succ_delta, opt_failed


def generate_ae(x_np, x_tf, model, sess, hps, min_over_all_classes, binary_search, batch_mode, verbose=False):
    time_start = time.time()
    batch_size = 25  # take smth that gives no remainder with 1000 and doesn't consume much memory
    n_examples = len(x_np)
    x_ae_batch = np.zeros_like(x_np)

    if batch_mode:
        n_batches = len(x_np) // batch_size
        f_vals_list, grad_vals_list = [], []
        for x_batch in x_np.reshape([n_batches, batch_size, hps.real_height, hps.real_width, hps.n_colors]):
            f_vals_batch = sess.run(model.get_logits(), feed_dict={x_tf: x_batch, model.flag_train: False,
                                                                   model.flag_take_grads: False})
            grad_vals_batch = sess.run(model.get_gradients(), feed_dict={x_tf: x_batch, model.flag_train: False,
                                                                         model.flag_take_grads: True})
            f_vals_list.append(f_vals_batch)
            grad_vals_list.append(grad_vals_batch)
        f_vals, grad_vals = np.vstack(f_vals_list), np.hstack(grad_vals_list)
    else:
        f_vals = sess.run(model.get_logits(), feed_dict={x_tf: x_np, model.flag_train: False,
                                                         model.flag_take_grads: False})
        grad_vals = sess.run(model.get_gradients(), feed_dict={x_tf: x_np, model.flag_train: False,
                                                               model.flag_take_grads: True})

    x_batch_flat = np.reshape(x_np, [n_examples, hps.n_input_real])  # transform images into long vectors
    # grad_vals = np.reshape(grad_vals, [hps.n_classes, n_examples, hps.n_input_real])  # transform gradients wrt images into gradients wrt long vector
    min_norms = np.zeros(n_examples)
    for i, x in enumerate(x_batch_flat):
        min_delta, min_norm = np.zeros(hps.n_input_real), np.inf
        top_classes = np.argsort(f_vals[i])[-4:][::-1]
        pred_class, other_top_classes = top_classes[0], top_classes[1:]
        if min_over_all_classes:
            other_classes = [cls for cls in range(hps.n_classes) if cls != pred_class]
        else:
            # other_classes = [np.random.choice(other_top_classes)]
            other_classes = other_top_classes[0:1]
        for k in other_classes:
            v = grad_vals[pred_class, i] - grad_vals[k, i]
            if verbose:
                print('avg grad diff: ', np.abs(v).mean())
            c = f_vals[i, k] - f_vals[i, pred_class]

            if binary_search:
                delta, opt_failed = ae_binary_search(model, sess, x_tf, x, v, c, hps.dataset, pred_class, hps.n_classes,
                                                     attacks.linear_approx_boxc_l2, verbose)
            else:
                c *= np.random.rand()
                delta, opt_failed = attacks.linear_approx_boxc_l2(x, v, c)

            delta_norm = np.linalg.norm(delta, 2)
            if delta_norm < min_norm and not opt_failed:
                min_norm = delta_norm
                min_delta = delta
        min_norms[i] = min_norm
        x_ae_batch[i] = np.reshape(x + min_delta, [hps.real_height, hps.real_width, hps.n_colors])
    if verbose:
        print("Avg ae L2 norm: {:.4f}, done in: {:.3f}".format(np.mean(min_norms[min_norms != np.inf]), time.time() - time_start))
    return x_ae_batch, min_norms

