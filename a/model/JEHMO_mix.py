import tensorflow as tf
import numpy as np
import keras
import tensorflow_probability as tfp

Uniform = tfp.distributions.Uniform


class JEHMO_mixup(keras.Model):
    def __init__(
            self,
            ld_lr=1.,  # Lavegning dynamics gradient scale.
            ld_std=1e-2,  # Lavegning dynamics noise std.
            ld_n=40,  # Lavegning dynamics number of steps.
            ood=False,  # Flag to train with ood points.
            od_n=25,  # ood number of steps.
            od_lr=1.,  # Gradient scale for ood points
            od_std=.1,  # Std of ood point noise
            od_l=.01,  # ood loss scaling
            n_warmup=500,  # Number of steps without training ood.
            name='JEHM',
            **kwargs
    ):
        super(JEHMO_mixup, self).__init__(name=name, **kwargs)
        # p(x_o) sampler parameters
        self.od_lr = od_lr
        self.od_std = od_std
        self.od_l = od_l
        self.od_n = od_n
        self.n_warmup = n_warmup
        self.ood = ood
        # p(x) sampler parameters
        self.ld_lr = ld_lr
        self.ld_n = ld_n
        print(f"Samples: {self.ld_n}")
        self.ld_std = ld_std
        # Control parameters
        self.n_epochs = 0

    def sample_ood(self, x):
        ood_x = tf.Variable(x)
        for i in range(self.od_n):
            with tf.GradientTape() as g1:
                g1.watch(ood_x)
                z = self(ood_x)
                lg_a = tf.reduce_logsumexp(z, axis=1, keepdims=True)
                e = tf.reduce_sum(tf.exp(z - lg_a) * (lg_a - z), axis=1, keepdims=True)
                energy_dx = g1.gradient(e, ood_x)
                ood_x.assign_add(self.od_lr * energy_dx + tf.random.normal(
                    ood_x.shape, mean=0.0, stddev=self.od_std, dtype=tf.float32))
        return ood_x

    def sample_q(self, bs=50, min_val=-5, max_val=5):
        x0 = Uniform(min_val, max_val).sample(bs)
        xk = tf.Variable(x0)
        for i in range(self.ld_n):
            with tf.GradientTape() as g1:
                g1.watch(xk)
                z = self(xk)
                lg_a = tf.reduce_logsumexp(z, axis=1, keepdims=True)
                e = tf.reduce_sum(tf.exp(z - lg_a) * (lg_a - z), axis=1, keepdims=True)
                energy_dx = g1.gradient(e, xk)
                xk.assign_add(-self.ld_lr * energy_dx +
                              tf.random.normal(xk.shape, stddev=self.ld_std))
        return xk

    def train_step(self, data):
        x, y = data
        loss = 0
        max_val = tf.math.reduce_max(x, axis=0)
        min_val = tf.math.reduce_min(x, axis=0)

        xk = self.sample_q(x.shape[0], min_val, max_val)

        if self.n_epochs > self.n_warmup and self.ood:
            ood_x = self.sample_ood(x)
            ood_z = self(ood_x)
            ood_entropy = self.od_l * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                tf.ones(y.shape) * 1 / y.shape[1], ood_z))
            loss += ood_entropy

        with tf.GradientTape() as g2:
            z = self(x)
            zk = self(xk)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, z))
            lg_a1 = tf.reduce_logsumexp(z, axis=1, keepdims=True)
            e1 = tf.reduce_sum(tf.exp(z - lg_a1) * (lg_a1 - z), axis=1, keepdims=True)
            lg_a2 = tf.reduce_logsumexp(zk, axis=1, keepdims=True)
            e2 = tf.reduce_sum(tf.exp(zk - lg_a2) * (lg_a2 - zk), axis=1, keepdims=True)
            loss += tf.reduce_mean(e1) - tf.reduce_mean(e2)

        loss_grads = g2.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(loss_grads, self.trainable_weights))
        self.compiled_metrics.update_state(y, z)
        return {m.name: m.result() for m in self.metrics}
