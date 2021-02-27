import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.layers import Dense
import math
import calibration as cal
import numpy as np
from tensorflow.python.keras.initializers import init_ops


class ECE_metrics(tfk.metrics.Metric):
    def __init__(self, name='ECE', num_of_bins=10):
        super().__init__()
        self.num_of_bins = num_of_bins
        self.acc_counts = self.add_weight('acc_counts', shape=(self.num_of_bins + 1), initializer=init_ops.zeros_initializer) #[0 for _ in range(num_bins+1)]
        self.conf_counts = self.add_weight('conf_counts', shape=(self.num_of_bins + 1), initializer=init_ops.zeros_initializer) #[0 for _ in range(num_bins+1)]
        self.counts = self.add_weight('counts', shape=(self.num_of_bins + 1), initializer=init_ops.zeros_initializer) #[0 for i in range(num_bins + 1)]
        self.ECE = self.add_weight(name="ece", initializer=init_ops.zeros_initializer)
        self.n = self.add_weight(name='n', initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)

        if not y_true.shape[0]:
            return
        self.n.assign_add(tf.cast(y_pred.shape[0], tf.float32))
        print(self.n)
        probabilities = tf.nn.softmax(y_pred, axis=1)
        confs = tf.reduce_max(probabilities, axis=1)
        preds = tf.argmax(probabilities, axis=1)
        labels = tf.argmax(y_true, axis=1)

        for (conf, pred, label) in zip(confs, preds, labels):
            bin_index = int(((conf * 100) // (100/self.num_of_bins)))
            if tf.equal(pred, label):
                self.acc_counts = tf.compat.v1.scatter_add(self.acc_counts, bin_index, 1.0)
            self.counts = tf.compat.v1.scatter_add(self.counts, bin_index, 1.0)
            self.conf_counts = tf.compat.v1.scatter_add(self.conf_counts, bin_index, tf.cast(conf, tf.float32))

    def result(self):
        if tf.equal(self.n, 0.0):
            return 0.0
        avg_acc = [float(0) if self.counts[i] == float(0) else self.acc_counts[i] / self.counts[i] for i in range(self.num_of_bins+1)]
        avg_conf = [float(0) if self.counts[i] == float(0) else self.conf_counts[i] / self.counts[i] for i in range(self.num_of_bins+1)]
        for i in range(self.num_of_bins+1):
            self.ECE.assign_add(tf.cast((self.counts[i] / self.n) * abs(avg_acc[i] - avg_conf[i]), tf.float32))
        return self.ECE

    def reset_states(self):
        self.acc_counts.assign(tf.zeros(self.num_of_bins + 1)) #self.add_weight('acc_counts', shape=(self.num_of_bins + 1), initializer="zeros") #[0 for _ in range(num_bins+1)]
        self.conf_counts.assign(tf.zeros(self.num_of_bins + 1)) #self.add_weight('conf_counts', shape=(self.num_of_bins + 1), initializer="zeros") #[0 for _ in range(num_bins+1)]
        self.counts.assign(tf.zeros(self.num_of_bins + 1)) #self.add_weight('counts', shape=(self.num_of_bins + 1), initializer="zeros") #[0 for i in range(num_bins + 1)]
        self.ECE.assign(0.0)
        self.n.assign(0.0)
        self.ECE.assign(0.0)


class OE_metrics(tfk.metrics.Metric):
    def __init__(self, name='OE', num_of_bins=10):
        super().__init__()
        self.num_of_bins = num_of_bins
        self.acc_counts = self.add_weight('acc_counts', shape=(self.num_of_bins + 1), initializer=init_ops.zeros_initializer) #[0 for _ in range(num_bins+1)]
        self.conf_counts = self.add_weight('conf_counts', shape=(self.num_of_bins + 1),initializer=init_ops.zeros_initializer) #[0 for _ in range(num_bins+1)]
        self.counts = self.add_weight('counts', shape=(self.num_of_bins + 1), initializer=init_ops.zeros_initializer) #[0 for i in range(num_bins + 1)]
        self.OE = self.add_weight(name="oe", initializer=init_ops.zeros_initializer)
        self.n = self.add_weight(name='n', initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)
        if not y_true.shape[0]:
            return
        self.n.assign_add(tf.cast(y_pred.shape[0], tf.float32))
        probabilities = tf.nn.softmax(y_pred, axis=1)
        confs = tf.reduce_max(probabilities, axis=1)
        preds = tf.argmax(probabilities, axis=1)
        labels = tf.argmax(y_true, axis=1)
        for (conf, pred, label) in zip(confs, preds, labels):
            bin_index = int(((conf * 100) // (100/self.num_of_bins)))
            if tf.equal(pred, label):
                tf.compat.v1.scatter_add(self.acc_counts, bin_index, 1.0)
            tf.compat.v1.scatter_add(self.counts, bin_index, 1.0)
            tf.compat.v1.scatter_add(self.conf_counts, bin_index, tf.cast(conf, tf.float32))
        return

    def result(self):
        if tf.equal(self.n, 0.0):
            return 0.0
        avg_acc = [float(0) if self.counts[i] == float(0) else self.acc_counts[i] / self.counts[i] for i in
                   range(self.num_of_bins+1)]
        avg_conf = [float(0) if self.counts[i] == float(0) else self.conf_counts[i] / self.counts[i] for i in
                    range(self.num_of_bins+1)]

        for i in range(self.num_of_bins+1):
            self.OE.assign_add(tf.cast((self.counts[i] / self.n) * (avg_conf[i] * (tf.maximum(avg_conf[i] - avg_acc[i], 0))), tf.float32))
        return self.OE

    def reset_states(self):
        self.acc_counts.assign(tf.zeros(self.num_of_bins + 1)) #self.add_weight('acc_counts', shape=(self.num_of_bins + 1), initializer="zeros") #[0 for _ in range(num_bins+1)]
        self.conf_counts.assign(tf.zeros(self.num_of_bins + 1)) #self.add_weight('conf_counts', shape=(self.num_of_bins + 1), initializer="zeros") #[0 for _ in range(num_bins+1)]
        self.counts.assign(tf.zeros(self.num_of_bins + 1)) #self.add_weight('counts', shape=(self.num_of_bins + 1), initializer="zeros") #[0 for i in range(num_bins + 1)]
        self.n.assign(0.0)
        self.OE.assign(0.0)


def compute_calibration_metrics(outputs, labels, num_bins=10, device='cuda'):
    """
    Computes the calibration metrics ECE and OE along with the acc and conf values
    :param num_bins: Taken from email correspondence and 100 is used
    :param net: trained network
    :param loader: dataloader for the dataset
    :param device: cuda or cpu
    :return: ECE, OE, acc, conf
    """
    labels = tf.cast(labels, tf.int32)
    outputs = tf.cast(outputs, tf.float32)
    print(outputs)
    acc_counts = [0 for _ in range(num_bins+1)]
    conf_counts = [0 for _ in range(num_bins+1)]
    overall_conf = []
    counts = [0 for i in range(num_bins + 1)]
    n = float(len(labels))
    '''
    n = float(len(loader.dataset))
    counts = [0 for i in range(num_bins+1)]
    net.eval()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
    '''
    probabilities = tf.nn.softmax(outputs, axis=1)
    confs = tf.reduce_max(probabilities, axis=1)
    preds = tf.argmax(probabilities, axis=1)
    labels = tf.argmax(labels, axis=1)
    for (conf, pred, label) in zip(confs, preds, labels):
        bin_index = int(((conf * 100) // (100/num_bins)))
        try:
            if pred == label:
                acc_counts[bin_index] += 1.0
            conf_counts[bin_index] += conf.cpu()
            counts[bin_index] += 1.0
            overall_conf.append(conf.cpu())
        except:
            print(bin_index, conf)
            raise AssertionError('Bin index out of range!')


    avg_acc = [0 if count == 0 else acc_count / count for acc_count, count in zip(acc_counts, counts)]
    avg_conf = [0 if count == 0 else conf_count / count for conf_count, count in zip(conf_counts, counts)]
    ECE, OE = 0, 0
    for i in range(num_bins):
        ECE += (counts[i] / n) * abs(avg_acc[i] - avg_conf[i])
        OE += (counts[i] / n) * (avg_conf[i] * (max(avg_conf[i] - avg_acc[i], 0)))

    return ECE, OE

y_true = np.array([[0, 0, 1],
                   [0, 1, 0],
                   [0, 0, 1],
                   [1, 0, 0],
                   [0, 0, 1],
                   [0, 1, 0],
                   [0, 0, 1]])
y_pred = np.array([[0.1, 0.9, 0.8],
                   [0.05, 0.95, 0],
                   [0.1, 0.8, 0.2],
                   [0.9, 0.1, 0.1],
                   [0.4, 0.8, 0.9],
                   [0.5, 0.7, 0.9],
                   [0.1, 0.1, 0.2],])
y_true = tf.convert_to_tensor(y_true)
ece, oe = compute_calibration_metrics(np.array(y_pred), np.array(y_true), num_bins=3, device='cuda')
ECE = ECE_metrics(num_of_bins=3)
OE = OE_metrics(num_of_bins=3)
ECE.update_state(y_true, y_pred)
ece1 = ECE.result().numpy()
OE.update_state(y_true, y_pred)
oe1 = OE.result().numpy()


probs = tf.nn.softmax(y_pred, axis=1).numpy()
labels = tf.argmax(y_true, axis=1).numpy()
a = cal.get_ece(probs, labels, num_bins=3)
print(a)
print(ece)
print(ece1)
print(oe1)


class CatgoricalTruePositives(tf.keras.metrics.Metric):
    def __init__(self, name='categorical_true_positives', **kwargs):
        super(CatgoricalTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred,axis=-1)
        y_true = tf.argmax(y_true,axis=-1)
        values = tf.equal(tf.cast(y_true, 'int32'), tf.cast(y_pred, 'int32'))
        values = tf.cast(values, 'float32')
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_states(self):
        self.true_positives.assign(0.)
