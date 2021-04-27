import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.layers import Dense
import math

class ManifoldMixup(tf.keras.Model):

    def __init__(self, hidden_layers=[16, 2, 16], output_shape=2):
        super(ManifoldMixup, self).__init__()

        for i, nodes in enumerate(hidden_layers):
            setattr(self, 'dense{}'.format(i + 1),
                    tf.keras.layers.Dense(nodes, activation=tf.nn.relu))

        self.final = tf.keras.layers.Dense(output_shape, activation="linear")

    def call(self, inputs, training=False):
        x1, x2, lam = inputs

        if training:
            k = tf.random.uniform((1,), minval=0, maxval=len(self.layers),
                                  dtype=tf.dtypes.int32, name="manifold_idx")[0]
        else:
            k = 0.0
            lam = 1.0
        x = x1
        if not training:
            # Denotes data mixup
            x = lam * x1 + (1 - lam) * x2

        mix = True
        for i, l in enumerate(self.layers):
            if not training: mix = False
            if mix:
                x1 = l(x1)
                x2 = l(x2)
                # Keep a copy of the mixed-up manifold
                x = lam * x1 + (1 - lam) * x2
            else:  # Proceed on mixed batch
                x = l(x)
        return x