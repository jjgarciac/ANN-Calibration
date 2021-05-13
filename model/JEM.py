import tensorflow as tf
import numpy as np
import keras
import tensorflow_probability as tfp
import torch as t
from .buffer import *
Uniform = tfp.distributions.Uniform

class JEM(keras.Model):
    def __init__(
        self,
            batch_size=32,
            reinit_freq=0.05,
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
            with_buffer_in=True,
            with_buffer_out=True,
            buffer_size=1000,
            **kwargs
    ):
        super(JEM, self).__init__(name=name, **kwargs)
        # p(x_o) sampler parameters
        self.buffer_size = buffer_size
        self.with_buffer_in = with_buffer_in
        self.with_buffer_out = with_buffer_out * (ood)
        self.reinit_freq = reinit_freq
        # other parameters
        self.bs = batch_size
        self.od_lr=od_lr
        self.od_std=od_std
        self.od_l=od_l
        self.od_n=int(od_n)
        self.n_warmup=n_warmup
        self.ood=ood
        # p(x) sampler parameters
        self.ld_lr=ld_lr
        self.ld_n=int(ld_n)
        print(f"Samples: {self.ld_n}")
        self.ld_std=ld_std
        # Control parameters
        self.n_epochs=0

    def sample_ood(self, x):
        if (not hasattr(self, 'replay_buffer_out')) and self.with_buffer_out:
            # replay buffer should be defined out of the class, if there is no buffer, return the x
                self.replay_buffer_out = x #get_buffer(self.buffer_size, x.shape[1])

        if self.with_buffer_out:
            init_samples, buffer_inds = self.sample_p_0_out(self.replay_buffer_out, x)
            #init_samples, buffer_inds = self.sample_p_0(self.replay_buffer_in, bs=x.shape[0], feat_size=x.shape[1])
        else:
            init_samples = x
            #init_samples = Uniform(-1, 1).sample(x.shape[0])

        ood_x = tf.Variable(x)
        for i in range(self.od_n):
            with tf.GradientTape() as g1:
                g1.watch(ood_x)
                z = self(ood_x)
                e = -tf.reduce_logsumexp(z, axis=1, keepdims=True)
                energy_dx = g1.gradient(e, ood_x)
                ood_x.assign_add(self.od_lr*energy_dx + tf.random.normal(
                    ood_x.shape,mean=0.0, stddev=self.od_std,dtype=tf.float32))
        # buffer
        final_samples = ood_x.value()  # .read_value()
        # update replay buffer
        if self.with_buffer_out and (len(self.replay_buffer_out) > 0):
            self.replay_buffer_out = tf.tensor_scatter_nd_update(self.replay_buffer_out,
                                                                buffer_inds[:, None],
                                                                final_samples)
        return ood_x

    def sample_q(self, bs=50, feat_size=100):
        if (not hasattr(self, 'replay_buffer_in')) and self.with_buffer_in:
            self.replay_buffer_in = get_buffer(self.buffer_size, feat_size, x=None)
        if self.with_buffer_in:
            init_samples, buffer_inds = self.sample_p_0(self.replay_buffer_in, bs=bs, feat_size=feat_size)
        else:
            init_samples = Uniform(-1, 1).sample(bs, feat_size)  # ?

        xk = tf.Variable(init_samples)
        for i in range(self.ld_n):
            with tf.GradientTape() as g1:
                g1.watch(xk)
                z = self(xk)
                e = -tf.reduce_logsumexp(z, axis=1, keepdims=True)
                energy_dx = g1.gradient(e, xk)
                xk.assign_add(-self.ld_lr*energy_dx + 
                        tf.random.normal(xk.shape, stddev=self.ld_std))
        # buffer
        final_samples = xk.value() #.read_value()
        # update replay buffer
        if self.with_buffer_in and (len(self.replay_buffer_in) > 0):
            self.replay_buffer_in = tf.tensor_scatter_nd_update(self.replay_buffer_in,
                                                                buffer_inds[:, None],
                                                                final_samples)
        return xk

    def sample_p_0(self, replay_buffer, bs, feat_size):
        if len(replay_buffer) == 0:
            return init_random(bs, feat_size), []
        buffer_size = len(replay_buffer)
        inds = np.random.randint(0, buffer_size, (bs,)) #t.randint(0, buffer_size, (bs,))
        # if cond, convert inds to class conditional inds
        buffer_samples = tf.gather(replay_buffer, inds)
        random_samples = init_random(bs, feat_size)
        choose_random = (tf.random.uniform((bs,), minval=0, maxval=1) < self.reinit_freq)[:, None]
        choose_random = tf.cast(choose_random, tf.float32)
        samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
        return samples, inds

    def sample_p_0_out(self, replay_buffer, x):
        if len(replay_buffer) == 0:
            return x, []
            #return init_random(x.shape[0], x.shape[1]), []
        buffer_size = len(replay_buffer)
        inds = np.random.randint(0, buffer_size, (x.shape[0],)) #t.randint(0, buffer_size, (bs,))
        # if cond, convert inds to class conditional inds
        buffer_samples = tf.gather(replay_buffer, inds)
        choose_random = (tf.random.uniform((x.shape[0],), minval=0, maxval=1) < self.reinit_freq)[:, None]
        choose_random = tf.cast(choose_random, tf.float32)
        samples = choose_random * x + (1 - choose_random) * buffer_samples
        #random_samples = init_random(x.shape[0], x.shape[1])
        #samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
        return samples, inds

    def train_step(self, data):
        x, y = data
        loss = 0
        
        xk = self.sample_q(x.shape[0], x.shape[1])

        '''
        if self.n_epochs > self.n_warmup and self.ood:
            ood_x = self.sample_ood(x)
            ood_z = self(ood_x)
            ood_entropy = self.od_l * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                tf.ones(y.shape)*1/y.shape[1], ood_z))
            loss += ood_entropy
        '''

        with tf.GradientTape() as g2:
            z = self(x)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, z))
            e1 = -tf.reduce_logsumexp(self(xk), axis=1, keepdims=True)
            e2 = -tf.reduce_logsumexp(z, axis=1, keepdims=True)
            loss += tf.reduce_mean(e2) - tf.reduce_mean(e1)
            if self.n_epochs > self.n_warmup and self.ood:
                ood_x = self.sample_ood(x)
                ood_z = self(ood_x)
                ood_entropy = self.od_l * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    tf.ones(y.shape) * 1 / y.shape[1], ood_z))
                loss += ood_entropy

        loss_grads = g2.gradient(loss, self.trainable_weights) 
        self.optimizer.apply_gradients(zip(loss_grads, self.trainable_weights))

        self.compiled_metrics.update_state(y, z)
        return {m.name: m.result() for m in self.metrics}
