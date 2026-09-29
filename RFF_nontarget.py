import math
import cleverhans
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.utils import to_categorical
import signal_representations as sr
from load_dataset import load_single_file
from tensorflow.keras import Model
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from signal_representations import wgn, eps2per


class TestMain:
    def test(self):
        # Load training and test data
        eps = 0.2
        norm = 2  # 2/np.inf

        psr_set = tf.range(-40, -9)

        # SNR_set = [55, 50, 45, 40, 35, 30, 25]
        # for snr in SNR_set:
        for psr in psr_set:
            tx_range = np.arange(30, 40, dtype=int)
            [data_test, label_test] = load_single_file('./dataset/test/usrp_20220915test.h5',
                                                       'data',
                                                       'label',
                                                       tx_range,
                                                       pkt_range=range(0, 100))
            label_test = label_test - tx_range[0]
            label_test2 = to_categorical(label_test, num_classes=10)

            model = tf.keras.models.load_model('GRU_Adam.h5', compile=False)
            data_test = sr.dspectrogram(data_test, win_len=256, crop_ratio=0.3)
            #
            data_test = data_test[:, :, :, 0]
            data_test = data_test.transpose(0, 2, 1)

            orignal = model.predict(data_test)
            orignal = orignal.argmax(axis=-1)
            y_pred2 = to_categorical(orignal, num_classes=10)
            clean_acc = accuracy_score(label_test2, y_pred2)

            b_size = 100
            x_fgm = []
            for i in range(int(len(data_test) / b_size)):
                x_fgm.append(
                    fast_gradient_method(model, data_test[i * b_size:i * b_size + b_size], eps, norm))

            x_fgm, data_per_adv_ideal_fgm = eps2per(np.concatenate(x_fgm, axis=0), data_test, psr)
            fgm = model.predict(x_fgm)
            fgm = fgm.argmax(axis=-1)
            y_pred_fgm2 = to_categorical(fgm, num_classes=10)
            fool_fgsm = accuracy_score(y_pred2, y_pred_fgm2)

            x_pgd = []
            for i in range(int(len(data_test) / b_size)):
                x_pgd.append(
                    projected_gradient_descent(model, data_test[i * b_size:i * b_size + b_size], eps, 0.01, 20, norm))
            # x_pgd = np.concatenate(x_pgd, axis=0)
            x_pgd, data_per_adv_ideal_pgd = eps2per(np.concatenate(x_pgd, axis=0), data_test, psr)
            pgd = model.predict(x_pgd)
            pgd = pgd.argmax(axis=-1)
            y_pred_pgd2 = to_categorical(pgd, num_classes=10)
            fool_pgd = accuracy_score(y_pred2, y_pred_pgd2)

            fgsm_fool = 1 - fool_fgsm
            pgd_fool = 1 - fool_pgd

            print(
                '-----------------------PSR =  %.4f' % psr)
            print(
                'test clean_acc on clean examples = %.4f' % clean_acc)
            print(
                ' fool_fgsm = %.4f, fool_pgd = %.4f' % (fgsm_fool, pgd_fool))

            data_test_addAWGN = wgn(data_test, psr)
            label_test_addAWGN = model.predict(data_test_addAWGN)
            label_test_addAWGN = label_test_addAWGN.argmax(axis=-1)
            y_pred_noisy = to_categorical(label_test_addAWGN, num_classes=10)
            fool_noisy = 1 - accuracy_score(y_pred2, y_pred_noisy)
            print(
                'test AWGN as same perturbation as psr on clean examples = %.4f' % fool_noisy)

            conf_mat = confusion_matrix(label_test, pgd)
            classes = np.arange(30, 40) - 30 + 1
            plt.figure(figsize=(5.8, 5))
            cm = conf_mat / 100
            sns.heatmap(cm, annot=True, annot_kws={"size": 13},
                        fmt='.2g', cmap='Blues',
                        cbar=False,
                        xticklabels=classes,
                        yticklabels=classes)

            plt.xlabel('Predicted label', fontsize=18)
            plt.ylabel('True label', fontsize=18)
            plt.xticks(size=18)
            plt.yticks(size=18)
            plt.savefig('GRUnew.pdf', bbox_inches='tight')
            plt.close()


if __name__ == '__main__':
    print("this is main method")
    TestMain.test('self')
