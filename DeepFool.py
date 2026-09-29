import numpy as np
from tensorflow.keras.models import Model
import copy
import tensorflow as tf
import matplotlib.pyplot as plt
import random
def deepfool(input, pretrained_model, overshoot=0.004, max_iter=50, num_classes=10):
    '''
    :param input: Whole dataset (n,input_shape)
    :param pretrained_model: feedforward function
    '''
    model = pretrained_model

    input_norm = tf.cast(input, tf.float32)
    # input_norm = input_norm[None, :, :, :]
    f_input = model(input_norm).numpy().flatten()

    I = (np.array(f_input)).flatten().argsort()[::-1]
    I = I[0:num_classes]
    label = I[0]
    # print(label, "label")
    input_shape = np.shape(input_norm)
    pert_input = copy.deepcopy(input_norm)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)
    loop_i = 0
    x = tf.Variable(pert_input)
    # fs = model(x)
    k_i = label
    def loss_func(logits, I, k):
        # return tf.nn.softmax_cross_entropy_with_logits(labels_pred=labels_pred, logits=logits)
        return logits[0, I[k]]
    while k_i == label and loop_i < max_iter:
        # print(loop_i)
        pert = np.inf
        # one_hot_label_0 = tf.one_hot(label, num_classes)
        with tf.GradientTape() as tape:
            tape.watch(x)
            fs = model(x)
            # loss_value = loss_func(one_hot_label_0, fs)
            loss_value = loss_func(fs, I, 0)
            # loss_value = fs[0,I[0]]
        # grad_orig = tape.gradient(fs[0, I[0]], x)
        grad_orig = tape.gradient(loss_value, x)
        for k in range(1, num_classes):
            # one_hot_label_k = tf.one_hot(I[k], num_classes)
            with tf.GradientTape() as tape:
                tape.watch(x)
                fs = model(x)
                # loss_value = loss_func(one_hot_label_k, fs)
                loss_value = loss_func(fs, I, k)

            # cur_grad = tape.gradient(fs[0, I[k]], x)
            cur_grad = tape.gradient(loss_value, x)
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).numpy()

            pert_k = abs(f_k) / (np.linalg.norm(tf.reshape(w_k, [-1]))+1e-4)
            if pert_k < pert:
                pert = pert_k
                w = w_k

        r_i = (pert + 1e-4) * w / (np.linalg.norm(w)+1e-4)
        r_tot = np.float32(r_tot + r_i)
        pert_input = input_norm + (1 + overshoot) * r_tot
        x = tf.Variable(pert_input)
        fs = model(x)
        k_i = np.argmax(np.array(fs).flatten())
        loop_i += 1
    r_tot = (1 + overshoot) * r_tot
    return r_tot, loop_i, label, k_i, np.asarray(pert_input)


def deepfool_target(input, pretrained_model, target, overshoot=0.002, max_iter=5, num_classes=10):
    '''
    :param input: Whole dataset (n,input_shape)
    :param pretrained_model: feedforward function
    '''

    def _loss_func(logits, I, k):
        # return tf.nn.softmax_cross_entropy_with_logits(labels_pred=labels_pred, logits=logits)
        return logits[0, I[k]]

    model = pretrained_model

    input_norm = tf.cast(input, tf.float32)
    # input_norm = input_norm[None, :, :, :]
    f_input = model(input_norm).numpy().flatten()

    I = (np.array(f_input)).flatten().argsort()[::-1]
    I = I[0:num_classes]
    I_g = np.arange(0, 10)
    label = I[0]

    input_shape = np.shape(input_norm)
    pert_input = copy.deepcopy(input_norm)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)
    loop_i = 0
    x = tf.Variable(pert_input)

    k_i = label

    loss_history = []
    while k_i != target and loop_i < max_iter:

        pert = np.inf
        # one_hot_label_0 = tf.one_hot(label, num_classes)
        with tf.GradientTape() as tape:
            tape.watch(x)
            fs = model(x)

            loss_value = _loss_func(fs, I, 0)

        grad_orig = tape.gradient(loss_value, x)
        with tf.GradientTape() as tape:
            tape.watch(x)
            fs = model(x)
             # loss_value = loss_func(one_hot_label_k, fs)
            loss_value = _loss_func(fs, I_g, target)
            loss_history.append(loss_value)

            cur_grad = tape.gradient(loss_value, x)
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I_g[target]] - fs[0, I[0]]).numpy()
            z = np.linalg.norm(tf.reshape(w_k, [-1]))
            pert_k = abs(f_k) / (np.linalg.norm(tf.reshape(w_k, [-1])) + 1e-32)
            if pert_k < pert:
                pert = pert_k
                w = w_k

        r_i = (pert + 1e-4) * w / (np.linalg.norm(w)+1e-32)
        r_tot = np.float32(r_tot + r_i)
        pert_input = input_norm + (1 + overshoot) * r_tot
        x = tf.Variable(pert_input)
        fs = model(x)
        k_i = np.argmax(np.array(fs).flatten())
        loop_i += 1

    r_tot = (1 + overshoot) * r_tot
    return r_tot, loop_i, label, k_i, target, np.asarray(pert_input)

