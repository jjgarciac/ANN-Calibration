import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.layers import Dense
import math
from metrics import *

def gauss_pdf(x, name=None):
    return (1/tf.sqrt(2*math.pi))*tf.math.exp((-tf.pow(x, 2))/2)



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
