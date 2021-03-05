import tensorflow as tf
import numpy as np
import keras
import tensorflow_probability as tfp

Uniform = tfp.distributions.Uniform

class JEM(keras.Model):
    def sample_q(self, bs=50, sgld_lr=.2, sfld_std=1e-2,
                 N=20, min_val=-5, max_val=5):
        x0 = Uniform(min_val, max_val).sample(bs)
        xk = tf.Variable(x0)
        T=1.
        for i in range(N):
            with tf.GradientTape() as g1:
                g1.watch(xk)
                z = self(xk)
                e = -T*tf.reduce_logsumexp(z, axis=1, keepdims=True)
                energy_dx = g1.gradient(e, xk)
                xk.assign_add(-sgld_lr*energy_dx + tf.random.normal(xk.shape, stddev=sfld_std))
        return xk

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        
        max_val = tf.math.reduce_max(x, axis=0)
        min_val = tf.math.reduce_max(x, axis=0)
        
        x_tensor = x
        ood_x = tf.Variable(x)
        ood_energy = 0
        ood_entropy = 0

        xk = self.sample_q(x.shape[0], min_val=min_val, max_val=max_val)
        currEpoch=2
        WARMUP=1
        N=25
        ALPHA=.2
        SIGMA=.1
        LAMBDA=.005 #To account for no warmup
        T=1.

        if currEpoch > WARMUP:
            for i in range(N):
                with tf.GradientTape() as g1:
                    g1.watch(ood_x)
                    z = self(ood_x)
                    e = -T*tf.reduce_logsumexp(z, axis=1, keepdims=True)
                    energy_dx = g1.gradient(e, ood_x)
                ood_x = ood_x + ALPHA * energy_dx + tf.random.normal(
                        x_tensor.shape,mean=0.0, stddev=SIGMA,dtype=tf.float32)

        with tf.GradientTape() as g2:
            z = self(x_tensor)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, z))
            e1 = -T*tf.reduce_logsumexp(self(xk), axis=1, keepdims=True)
            e2 = -T*tf.reduce_logsumexp(z, axis=1, keepdims=True)
            loss += tf.reduce_mean(e2) - tf.reduce_mean(e1)

            if currEpoch > WARMUP:
                ood_z = self(ood_x)
                ood_entropy = LAMBDA * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    tf.ones(y.shape)*1/y.shape[1], ood_z))
                ood_energy = -T*tf.reduce_logsumexp(ood_z, axis=1, keepdims=True)
                ood_energy = tf.reduce_mean(ood_energy)
                ood_loss = ood_entropy
                loss += ood_loss

        loss_grads = g2.gradient(loss, self.trainable_weights) #Check what the difference is with trainable variables

        self.optimizer.apply_gradients(zip(loss_grads, self.trainable_weights))
        

        #loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        #trainable_vars = self.trainable_variables
        # Update weights
        #self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, z)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
