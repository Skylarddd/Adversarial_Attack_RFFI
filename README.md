# Adversarial Attacks Against Deep Learning-Based RF Fingerprint Identification

## 1. System

The experiments are conducted using a **LoRa-based RFFI testbed**.

The RFFI system consists of:

* A USRP N210 software-defined radio (SDR) receiver
* Multiple commercial off-the-shelf (COTS) LoRa transmitters
* Deep-learning-based RFFI classifiers

The evaluated neural-network architectures include:

* **CNN**
* **LSTM**
* **GRU**

These models are trained to classify signals according to their transmitting devices. The trained classifiers are subsequently used as victim models for adversarial attack experiments.

## 2. Repository Structure

```text

├── main_train.py
├── main_test.py
├── RFF_nontarget.py
├── Result_all_target_attack_plot.py
├── UAP_psr.py
├── UAP_Surrogate_Device.py
└── Mix_Scenarios.py
```

### `main_train.py`

Trains the RFFI classification models.

The trained models are used as victim models for subsequent adversarial attack experiments.

Supported model architectures include:

* CNN
* LSTM
* GRU

---

### `main_test.py`

Evaluates the performance of the trained RFFI models on clean test samples.

This provides the baseline classification performance before adversarial perturbations are introduced.

---

### `RFF_nontarget.py`

Generates adversarial perturbations for **non-targeted attacks**.

Implemented attacks include:

* **FGSM**
* **PGD**

---

### `Result_all_target_attack_plot.py`

Processes and visualizes the experimental results for **targeted PGD attacks**.

The script is mainly used for generating figures and comparing targeted attack performance under different experimental settings.

---

### `UAP_psr.py`

Implements **Universal Adversarial Perturbation (UAP)** experiments.

The generated UAP is applied to multiple signal samples to investigate whether a universal perturbation can effectively degrade the performance of the RFFI classifier.

---

### `UAP_Surrogate_Device.py`

Implements **surrogate-device-based UAP attacks**.

The perturbation is generated using a surrogate device and then transferred to the victim device.

This experiment is designed to approximate a more realistic attack scenario where the adversary has limited prior knowledge of the victim system.

---

### `Mix_Scenarios.py`

Evaluates adversarial attacks under multiple practical scenarios, including:

* **Real-time attacks**
* **Surrogate-device attacks**
* **Cross-model transferability**
* **Mixed attack scenarios**

This script brings together several attack settings to evaluate the practical effectiveness and robustness of adversarial attacks against RFFI systems.

---

## 2. Requirements

The code is implemented primarily in Python and uses deep-learning and scientific-computing libraries.

Typical dependencies include:

```text
Python
Tensorflow
SciPy
Matplotlib
scikit-learn...
```

Please refer to the environment.yaml file for the required environment.


## 3. Citation

If you use this code or reproduce the experiments, please cite:

```bibtex
@article{ma2026adversarial,
  author  = {Jie Ma and Junqing Zhang and Guanxiong Shen and
             Alan Marshall and Chip-Hong Chang},
  title   = {Adversarial Attacks Against Deep Learning-Based
             Radio Frequency Fingerprint Identification},
  journal = {IEEE Transactions on Mobile Computing},
  volume  = {25},
  number  = {6},
  pages   = {7831--7844},
  year    = {2026},
  doi     = {10.1109/TMC.2025.3646257}
}
```