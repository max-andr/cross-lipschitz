"""
Different neural network architectures are defined here as
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages


class BasicNN:
    """
    An abstract class for a Neural Network model.
    """
    def __init__(self, flag_train, flag_take_grads, hps):
        self.flag_train = flag_train
        self.flag_take_grads = flag_take_grads
        self.hps = hps
        self.activation = hps.activation
        self.extra_train_ops = []
        self.f, self.grads_f_x, self.hidden1 = None, None, None

    def get_logits(self):
        return self.f

    def get_gradients(self):
        return self.grads_f_x

    def run_logits(self, x_np, x_tf, sess):
        batch_size = len(x_np)
        if len(x_np.shape) == 2:
            x_np = np.reshape(x_np, [batch_size, self.hps.real_height, self.hps.real_width, self.hps.n_colors])

        f_vals = sess.run(self.f, feed_dict={x_tf: x_np, self.flag_train: False, self.flag_take_grads: False})
        return f_vals

    def build_graph(self, x):
        raise NotImplementedError


class ResNet(BasicNN):
    def __init__(self, flag_train, flag_take_grads, hps):
        """ResNet constructor.
        ResNet model. Based on Ritchie Ng ResNet model: https://github.com/ritchieng/resnet-tensorflow
        Related papers:
        https://arxiv.org/pdf/1512.03385v1.pdf - main paper
        https://arxiv.org/pdf/1603.05027v2.pdf - Identity Mappings in Deep Residual Networks
        https://arxiv.org/pdf/1605.07146v1.pdf - wide residual networks

        Args:
          flag_train: tf.bool() which is True when we run the comp. graph for training, False for testing
          flag_take_grads: tf.bool() which is True when we need to take gradients to calculate the robustness
          hps: object with hyperparameters.
        """
        super().__init__(flag_train, flag_take_grads, hps)

        if self.hps.use_bottleneck:
            self.res_func = self._bottleneck_residual
            self.filters = [16, 64, 128, 256]
        else:
            self.res_func = self._residual
            self.filters = [16, 16, 32, 64]
            # Wide ResNets are more memory efficient than very deep residual network
            # filters = [16, 160, 320, 640] and Update hps.n_resid_units to 9

    @staticmethod
    def _stride_arr(stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def build_graph(self, x):
        """
        Build the core model within the graph.
          x: Batches of images. [batch_size, image_size, image_size, 3]
        """
        filters = self.filters
        res_func = self.res_func

        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]

        with tf.variable_scope('init'):
            x = self._conv('init_conv', x, 3, 3, 16, self._stride_arr(1))

        # r res layers (16 filters)
        with tf.variable_scope('unit_1_0'):
            x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]), activate_before_residual[0])
        for i in range(1, self.hps.n_resid_units):
            with tf.variable_scope('unit_1_%d' % i):
                x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

        # r res layers (32 filters)
        with tf.variable_scope('unit_2_0'):
            x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]), activate_before_residual[1])
        for i in range(1, self.hps.n_resid_units):
            with tf.variable_scope('unit_2_%d' % i):
                x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

        # r res layers (64 filters)
        with tf.variable_scope('unit_3_0'):
            x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]), activate_before_residual[2])
        for i in range(1, self.hps.n_resid_units):
            with tf.variable_scope('unit_3_%d' % i):
                x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

        with tf.variable_scope('unit_last'):
            x = self._batch_norm('final_bn', x)
            x = self.activation(x)
            x = self._global_avg_pool(x)

        with tf.variable_scope('logit'):
            f = self._fully_connected(x)

        grad_matrix_list = [tf.reshape(tf.gradients(f[:, k], self.x)[0], [-1, self.hps.n_input_real])
                            for k in range(self.hps.n_classes)]
        grads_f_x = tf.stack(grad_matrix_list)
        self.f, self.grads_f_x = f, grads_f_x
        return f, grads_f_x

    def _batch_norm(self, name, x):
        """Batch normalization."""
        def moments_train(moving_mean, moving_variance, extra_train_ops):
            mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
            extra_train_ops.append(moving_averages.assign_moving_average(moving_mean, mean, 0.9))
            extra_train_ops.append(moving_averages.assign_moving_average(moving_variance, variance, 0.9))
            return mean, variance

        def moments_test(moving_mean, moving_variance, flag_take_grads):
            return tf.cond(flag_take_grads,
                           lambda: tf.nn.moments(x, [0, 1, 2], name='moments'),
                           lambda: (moving_mean, moving_variance))

        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable('beta', params_shape, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable('gamma', params_shape, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32))

            moving_mean = tf.get_variable('moving_mean', params_shape, tf.float32,
                                          initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
            moving_variance = tf.get_variable('moving_variance', params_shape, tf.float32,
                                              initializer=tf.constant_initializer(1.0, tf.float32), trainable=False)
            mean, variance = tf.cond(self.flag_train,
                                     lambda: moments_train(moving_mean, moving_variance, self.extra_train_ops),
                                     lambda: moments_test(moving_mean, moving_variance, self.flag_take_grads))

            # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
            y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())
            return y

    def _residual(self, x, in_filter, out_filter, stride, activate_before_residual=False):
        """Residual unit with 2 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = self._batch_norm('init_bn', x)
                x = self.activation(x)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self.activation(x)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self.activation(x)
            if 'dropout' in self.hps.reg_type:
                x = tf.cond(self.flag_train, lambda: tf.nn.dropout(x, self.hps.keep_hidden), lambda: x)
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(
                    orig_x, [[0, 0], [0, 0], [0, 0],
                             [(out_filter - in_filter) // 2,
                              (out_filter - in_filter) // 2]])
            x += orig_x

        tf.logging.info('image after unit %s', x.get_shape())
        return x

    def _bottleneck_residual(self, x, in_filter, out_filter, stride,
                             activate_before_residual=False):
        """Bottleneck resisual unit with 3 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('common_bn_relu'):
                x = self._batch_norm('init_bn', x)
                x = self.activation(x)
                orig_x = x
        else:
            with tf.variable_scope('residual_bn_relu'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self.activation(x)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 1, in_filter, out_filter / 4, stride)

        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self.activation(x)
            x = self._conv('conv2', x, 3, out_filter / 4, out_filter / 4, [1, 1, 1, 1])

        with tf.variable_scope('sub3'):
            x = self._batch_norm('bn3', x)
            x = self.activation(x)
            x = self._conv('conv3', x, 1, out_filter / 4, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = self._conv('project', orig_x, 1, in_filter, out_filter, stride)
            x += orig_x

        tf.logging.info('image after unit %s', x.get_shape())
        return x

    @staticmethod
    def _conv(name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable('weights', [filter_size, filter_size, in_filters, out_filters], tf.float32,
                                     initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
            return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    def _fully_connected(self, x):
        """FullyConnected layer for final output."""
        n_last_conv = self.filters[-1]  # we simply take the number of last feature maps, because of global average pooling
        x = tf.reshape(x, [self.hps.n_ex, n_last_conv])
        w = tf.get_variable('weights', [n_last_conv, self.hps.n_classes], initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [self.hps.n_classes], initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    @staticmethod
    def _global_avg_pool(x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])


class MLP1Layer(BasicNN):
    def __init__(self, flag_train, flag_take_grads, hps):
        """
        Fully-connected 1 hidden layer neural network.

        Args:
          flag_train: tf.bool() which is True when we run the comp. graph for training, False for testing
          flag_take_grads: tf.bool() which is True when we need to take gradients to calculate the robustness
          hps: object with hyperparameters.
        """

        super().__init__(flag_train, flag_take_grads, hps)
        self.w = {'h1': tf.Variable(tf.sqrt(1.0 / (self.hps.n_input_real * hps.n_hidden1)) *
                                    tf.random_normal([self.hps.n_input_real, self.hps.n_hidden1], seed=self.hps.r_seed), name='weights'),
                  'out': tf.Variable(tf.sqrt(1.0 / (self.hps.n_classes * self.hps.n_hidden1)) *
                                     tf.random_normal([self.hps.n_hidden1, self.hps.n_classes], seed=self.hps.r_seed), name='weights')}
        self.b = {'h1':  tf.Variable(0.1 * tf.ones([self.hps.n_hidden1]), name='biases'),
                  'out': tf.Variable(0.1 * tf.ones([self.hps.n_classes]), name='biases')}

    def get_hidden1(self):
        return self.hidden1

    def get_weight_dicts(self):
        return self.w, self.b

    def build_graph(self, x):
        """Build the core model within the graph."""
        self.x = x
        with tf.variable_scope('hidden1'):
            if 'dropout' in self.hps.reg_type:
                x = tf.cond(self.flag_train, lambda: tf.nn.dropout(x, self.hps.keep_input), lambda: x)
            hidden1 = tf.matmul(x, self.w['h1']) + self.hps.bias_flag * self.b['h1']
            hidden1_sigm = self.hps.activation(hidden1)

        with tf.variable_scope('output'):
            if 'dropout' in self.hps.reg_type:
                hidden1_sigm = tf.cond(self.flag_train, lambda: tf.nn.dropout(hidden1_sigm, self.hps.keep_hidden),
                                       lambda: hidden1_sigm)
            f = tf.matmul(hidden1_sigm, self.w['out']) + self.hps.bias_flag * self.b['out']

        grad_matrix_list = [tf.gradients(f[:, k], self.x)[0] for k in range(self.hps.n_classes)]
        grads_f_x = tf.stack(grad_matrix_list)

        self.f, self.grads_f_x, self.hidden1 = f, grads_f_x, hidden1
        return f, grads_f_x, hidden1


class CNNBasic(BasicNN):
    def __init__(self, flag_train, flag_take_grads, hps):
        """
        A basic CNN architecture.

        Args:
          flag_train: tf.bool() which is True when we run the comp. graph for training, False for testing
          flag_take_grads: tf.bool() which is True when we need to take gradients to calculate the robustness
          hps: object with hyperparameters.
        """
        super().__init__(flag_train, flag_take_grads, hps)

    def build_graph(self, x):
        n_filters_conv1 = 96  # 96
        n_filters_conv2 = 128  # 128
        n_hidden = 1024  # 256
        w = {'conv1': self.weight_variable([5, 5, self.hps.n_colors, n_filters_conv1], self.hps),
             'conv2': self.weight_variable([5, 5, n_filters_conv1, n_filters_conv2], self.hps),
             'fc1':   self.weight_variable([self.hps.real_height // 4 * self.hps.real_width // 4 * n_filters_conv2, n_hidden], self.hps),
             'fc2':   self.weight_variable([n_hidden, self.hps.n_classes], self.hps)}
        b = {'conv1': self.bias_variable([n_filters_conv1]),
             'conv2': self.bias_variable([n_filters_conv2]),
             'fc1':   self.bias_variable([n_hidden]),
             'fc2':   self.bias_variable([self.hps.n_classes])}

        if 'dropout' in self.hps.reg_type:
            X_input = tf.cond(self.flag_train, lambda: tf.nn.dropout(x, self.hps.keep_input), lambda: X_input)
        # Convolutional layer 1
        h_conv1 = self.hps.activation(self.conv2d(X_input, w['conv1']) + self.hps.bias_flag * b['conv1'])
        h_pool1 = self.max_pool_2x2(h_conv1)
        if 'dropout' in self.hps.reg_type:
            h_pool1 = tf.cond(self.flag_train, lambda: tf.nn.dropout(h_pool1, self.hps.keep_conv), lambda: h_pool1)
        # h_pool1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

        # Convolutional layer 2
        h_conv2 = self.hps.activation(conv2d(h_pool1, w['conv2']) + self.hps.bias_flag * b['conv2'])
        h_pool2 = self.max_pool_2x2(h_conv2)
        if 'dropout' in self.hps.reg_type:
            h_pool2 = tf.cond(self.flag_train, lambda: tf.nn.dropout(h_pool2, self.hps.keep_conv), lambda: h_pool2)
        # h_pool2 = tf.nn.lrn(h_pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        h_pool2_flat = tf.reshape(h_pool2, [-1, tf.shape(w['fc1'])[0]])

        # Fully connected layer 1
        h_fc1 = self.hps.activation(tf.matmul(h_pool2_flat, w['fc1']) + self.hps.bias_flag * b['fc1'])
        if 'dropout' in self.hps.reg_type:
            h_fc1 = tf.cond(self.flag_train, lambda: tf.nn.dropout(h_fc1, self.hps.keep_hidden), lambda: h_fc1)

        # Fully connected layer 2 (Output layer)
        f = tf.matmul(h_fc1, w['fc2']) + self.hps.bias_flag * b['fc2']
        return f

    @staticmethod
    def weight_variable(shape, hps):
        return hps.scale_init * tf.Variable(tf.random_normal(shape, stddev=1))

    @staticmethod
    def bias_variable(shape):
        return tf.Variable(tf.constant(0.0, shape=shape))

    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')


class CNNAdvanced(BasicNN):
    def __init__(self, flag_train, flag_take_grads, hps):
        """
        A more advanced CNN archicture, which is somewhat deeper than CNNBasic.

        Args:
          flag_train: tf.bool() which is True when we run the comp. graph for training, False for testing
          flag_take_grads: tf.bool() which is True when we need to take gradients to calculate the robustness
          hps: object with hyperparameters.
        """
        super().__init__(flag_train, flag_take_grads, hps)

    def build_graph(self, x):
        n_filters_conv1 = 96
        n_filters_conv2 = 128
        n_filters_conv3 = 256
        n_hidden1 = 1024
        n_hidden2 = 1024
        w = {'conv1':  self.weight_variable([5, 5, self.hps.n_colors, n_filters_conv1], self.hps),
             'conv2':  self.weight_variable([5, 5, self.hps.n_filters_conv1, n_filters_conv2], self.hps),
             'conv3':  self.weight_variable([5, 5, n_filters_conv2, n_filters_conv3], self.hps),
             'fc1':    self.weight_variable([self.hps.real_height // 4 * self.hps.real_width // 4 * n_filters_conv2, n_hidden1], self.hps),
             'fc2':    self.weight_variable([n_hidden1, n_hidden2], self.hps),
             'fc_out': self.weight_variable([n_hidden2, self.hps.n_classes], self.hps)}
        b = {'conv1':  self.bias_variable([n_filters_conv1]),
             'conv2':  self.bias_variable([n_filters_conv2]),
             'conv3':  self.bias_variable([n_filters_conv3]),
             'fc1':    self.bias_variable([self.hps.n_hidden1]),
             'fc2':    self.bias_variable([n_hidden2]),
             'fc_out': self.bias_variable([self.hps.n_classes])}

        # Convolutional layer 1
        h_conv1 = self.hps.activation(self.conv2d(x, w['conv1']) + self.hps.bias_flag * b['conv1'])
        h_pool1 = self.max_pool_2x2(h_conv1)

        # Convolutional layer 2
        h_conv2 = self.hps.activation(self.conv2d(h_pool1, w['conv2']) + self.hps.bias_flag * b['conv2'])
        h_pool2 = self.max_pool_2x2(h_conv2)

        # Convolutional layer 3
        h_conv3 = self.hps.activation(self.conv2d(h_pool2, w['conv3']) + self.hps.bias_flag * b['conv3'])
        h_pool3 = self.max_pool_2x2(h_conv3)

        h_pool3_flat = tf.reshape(h_pool3, [-1, tf.shape(w['fc1'])[0]])

        # Fully connected layer 1
        h_fc1 = self.hps.activation(tf.matmul(h_pool3_flat, w['fc1']) + self.hps.bias_flag * b['fc1'])
        if 'dropout' in self.hps.reg_type:
            h_fc1 = tf.cond(self.flag_train, lambda: tf.nn.dropout(h_fc1, self.hps.keep_hidden), lambda: h_fc1)

        # Fully connected layer 2
        h_fc2 = self.hps.activation(tf.matmul(h_fc1, w['fc2']) + self.hps.bias_flag * b['fc2'])
        if 'dropout' in self.hps.reg_type:
            h_fc2 = tf.cond(self.flag_train, lambda: tf.nn.dropout(h_fc2, self.hps.keep_hidden), lambda: h_fc2)

        # Fully connected layer 2 (Output layer)
        f = tf.matmul(h_fc2, w['fc_out']) + self.hps.bias_flag * b['fc_out']
        return f

    @staticmethod
    def weight_variable(shape, hps):
        return hps.scale_init * tf.Variable(tf.random_normal(shape, stddev=1))

    @staticmethod
    def bias_variable(shape):
        return tf.Variable(tf.constant(0.0, shape=shape))

    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')


