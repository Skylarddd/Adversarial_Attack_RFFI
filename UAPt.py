import numpy as np
import tensorflow as tf
import cleverhans
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
import random
from DeepFool import deepfool, deepfool_target
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from signal_representations import compute_power


def proj_lp(v, xi, p):
    # Project on the lp ball centered at 0 and of radius xi

    # SUPPORTS only p = 2 and p = Inf for now
    if p == 2:
        v = v * min(1, xi / np.linalg.norm(v.flatten(1)))
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
        raise ValueError('Values of p different from 2 and Inf are currently not supported...')

    return v


def universal_perturbation(dataset, f, delta=0.2, max_iter_uni=np.inf, xi=0.2, p=np.inf, num_classes=10, overshoot=0.12,
                           max_iter_df=5, target: int = None):
    """
    :param dataset: Images of size MxHxWxC (M: number of images)
    :param f: feedforward function (input: images, output: values of activation BEFORE softmax).
    :param grads: gradient functions with respect to input (as many gradients as classes).
    :param delta: controls the desired fooling rate (default = 80% fooling rate)
    :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = np.inf)
    :param xi: controls the l_p magnitude of the perturbation (default = 10)
    :param p: norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)
    :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
    :param max_iter_df: maximum number of iterations for deepfool (default = 10)
    :return: the universal perturbation.
    """
    # vset = np.empty(shape=(500,64,62,1))
    # dataset_perturbed = np.empty(shape=(500,64,62,1))
    v = 0
    attack_success_rate = 0.0
    num_images = dataset.shape[0]  # The images should be stacked ALONG FIRST DIMENSION
    itr = 0
    while attack_success_rate < 1 - delta and itr < max_iter_uni:

        print('Starting pass number', itr)

        # Go through the data set and compute the perturbation increments sequentially
        for m in range(0, num_images):
            cur_img = dataset[m:(m + 1), :, :]
            # cur_img = dataset[m:(m + 1), :, :, :]
            # print('The', m, 'st End')
            if int(np.argmax(np.array(f(cur_img)).flatten())) == int(np.argmax(np.array(f(cur_img + v)).flatten())):
                if target == None:
                    dr, iter, label, k_i, pert_image = deepfool(cur_img + v, f, num_classes=num_classes,
                                                                overshoot=overshoot, max_iter=max_iter_df)
                # be classified as target+1, range from 0 to 9
                else:
                    dr, iter, label, k_i, target, pert_image = deepfool_target(cur_img + v, f, target=target,
                                                                               num_classes=num_classes,
                                                                               overshoot=overshoot,
                                                                               max_iter=max_iter_df)
                # Make sure it converged...
                if iter < max_iter_df - 1:
                    v = v + dr
                    # Project on l_p ball
                    v = proj_lp(v, xi, p)

        itr = itr + 1
        dataset_perturbed = dataset + v
        # v = v.repeat(1000, axis=0)
        np.save('perturbation_v=', v)
        # dataset_perturbed_var = np.var(dataset_perturbed)
        # Perturb the dataset with computed perturbation
        label_all_ONEHOT = f.predict(dataset)
        label_all_ONEHOT = label_all_ONEHOT.argmax(axis=-1)
        label_all = to_categorical(label_all_ONEHOT, num_classes=10)
        k_i_all_ONEHOT = f.predict(dataset_perturbed)
        k_i_all_ONEHOT = k_i_all_ONEHOT.argmax(axis=-1)
        k_i_all = to_categorical(k_i_all_ONEHOT, num_classes=10)

        dataset_perturbed_power = compute_power(dataset_perturbed)
        data_set_power = compute_power(dataset)
        psr_uap = abs(dataset_perturbed_power - data_set_power) / data_set_power
        print('psr_uap = ', psr_uap)

        conf_mat = confusion_matrix(label_all_ONEHOT, k_i_all_ONEHOT)
        classes = np.arange(30, 40) - 30 + 1
        # fooling_rate = float(np.sum(label_all != k_i_all) / float(num_images))
        if target == None:
            attack_success_rate = 1 - accuracy_score(label_all, k_i_all)
        else:
            target_all_ONEHOT = np.full((num_images, 1), target)
            target_all = to_categorical(target_all_ONEHOT, num_classes=10)
            attack_success_rate = accuracy_score(target_all, k_i_all)
        print('Attack Success Rate = ', attack_success_rate)

        plt.figure(figsize=(5.8, 5))
        sns.heatmap(conf_mat, annot=True, annot_kws={"size": 13},
                    fmt='d', cmap='Blues',
                    cbar=False,
                    xticklabels=classes,
                    yticklabels=classes)

        plt.xlabel('Predicted label', fontsize=18)
        plt.ylabel('True label', fontsize=18)
        plt.xticks(size=18)
        plt.yticks(size=18)
        plt.savefig("UAP_result_{}.pdf".format(itr), bbox_inches='tight')
        plt.close()

    return dataset_perturbed, v
