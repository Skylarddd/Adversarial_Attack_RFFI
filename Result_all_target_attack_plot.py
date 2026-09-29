import math
import cleverhans
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import signal_representations as sr
from load_dataset import load_single_file, plt_sns_target, plot_confusion_matrix
from absl import app, flags
from tensorflow.keras import Model
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from signal_representations import calculate_labeltotarget_rate, eps2per


class TestMain:
    def test(self):
        # Load training and test data
        pgd_acc_target_all = []
        eps_set = [2]
        for eps in eps_set:
            target_set = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            for target in target_set:
                targeted = target - 1
                targeted_label = tf.constant(targeted, shape=(1000,), dtype=tf.int64)
                # target = 3
                tx_range = np.arange(30, 40, dtype=int)
                # [data_test, label_test] = load_single_file('./dataset/test/usrp_20221227test.h5',
                [data_test, label_test] = load_single_file('./dataset/test/usrp_20221226test.h5',
                                                           'data',
                                                           'label',
                                                           tx_range,
                                                           pkt_range=range(0, 100))
                # data_test = awgn(data_test, [15])
                label_test = label_test - tx_range[0]
                label_test2 = to_categorical(label_test, num_classes=10)
                model = tf.keras.models.load_model('GRU_model_1130_Adam_0129.h5', compile=False)

                # Evaluate on clean and adversarial data
                data_test = sr.dspectrogram(data_test, win_len=256, crop_ratio=0.3)

                data_test = data_test[:, :, :, 0]
                data_test = data_test.transpose(0, 2, 1)

                orignal = model.predict(data_test)
                orignal = orignal.argmax(axis=-1)
                y_pred2 = to_categorical(orignal, num_classes=10)
                clean_acc = accuracy_score(label_test2, y_pred2)

                # x_pgd_target, v = universal_perturbation(data_test, model, target=targeted)
                x_pgd_target = projected_gradient_descent(model, data_test, eps, 0.1, 100, np.inf, y=targeted_label,
                                                          targeted=True)
                x_pgd_target, data_per_adv_ideal = eps2per(x_pgd_target, data_test, -5)
                pgd_target = model.predict(x_pgd_target)
                pgd_target = pgd_target.argmax(axis=-1)
                y_pred_pgd_target2 = to_categorical(pgd_target, num_classes=10)

                acc_pgd_target = accuracy_score(label_test2, y_pred_pgd_target2)
                fool_pgd_target = accuracy_score(y_pred2, y_pred_pgd_target2)
                print(
                    'test clean_acc on clean examples = %.4f' % clean_acc)
                print(
                    'test pgd_target_acc on pgd_target examples = %.4f' % acc_pgd_target)
                print(
                    'test pgd_target_fool on pgd_target examples = %.4f' % fool_pgd_target)
                plt_sns_target(label_test, pgd_target, acc_pgd_target, target, eps, 'pgd_target')
                plt.close()
                pgd_target = pgd_target.reshape(1000, 1)
                i = 0
                while i < 1000:
                    pgd_acc_target = calculate_labeltotarget_rate(pgd_target[i:i + 100], targeted, 100)
                    pgd_acc_target_all.append(pgd_acc_target)
                    i = i + 100

            pgd_acc_target_all = np.array(pgd_acc_target_all)
            pgd_acc_target_all = pgd_acc_target_all.reshape(10, 10).T
            plot_confusion_matrix(pgd_acc_target_all, normalize=False,
                                  title='Target attack on each device')


if __name__ == '__main__':
    print("this is main method")
    TestMain.test('self')
