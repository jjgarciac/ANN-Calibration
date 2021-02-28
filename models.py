import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.layers import Dense
import math
from metrics import *

def gauss_pdf(x, name=None):
    return (1/tf.sqrt(2*math.pi))*tf.math.exp((-tf.pow(x, 2))/2)

class ManifoldMixup(tf.keras.Model):

    def __init__(self, hidden_layers=[16,2,16], output_shape=2):
        super(ManifoldMixup, self).__init__()
        
        for i, nodes in enumerate(hidden_layers):
            setattr(self, 'dense{}'.format(i+1),
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
        if k == 0:
            # Denotes data mixup
            x = lam*x1 + (1-lam)*x2
        
        mix = True
        for i,l in enumerate(self.layers):
            if (i+1) > k: mix=False
            if mix:
                x1 = l(x1)
                x2 = l(x2)
                # Keep a copy of the mixed-up manifold
                x = lam*x1 + (1-lam)*x2
            else: # Proceed on mixed batch
                x = l(x)
        return x

def build_model(in_shape, out_shape, manifold_mixup=False):

  if manifold_mixup:
    model = ManifoldMixup(hidden_layers=[64,64,64,64], output_shape=out_shape)
  else:
    model = tfk.Sequential([
        Dense(128, activation='relu', input_shape=(in_shape,)),
        Dense(128, activation='relu'),
        Dense(out_shape)
    ])

  optimizer = tfk.optimizers.Adam()
  loss = tfk.losses.CategoricalCrossentropy(from_logits=True)

  metrics = [ECE_metrics(name='ECE', num_of_bins=10),
             OE_metrics(name='OE', num_of_bins=10),
             tfk.metrics.CategoricalAccuracy('accuracy', dtype=tf.float32),]
  model.compile(optimizer=optimizer, 
                loss=loss, 
                metrics=metrics,
                run_eagerly=True)
  
  return model
