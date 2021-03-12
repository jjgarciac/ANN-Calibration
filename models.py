import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.layers import Dense
import math
from metrics import *
from model import *
import keras #Try and remove this import
from model import JEM

def gauss_pdf(x, name=None):
    return (1/tf.sqrt(2*math.pi))*tf.math.exp((-tf.pow(x, 2))/2)

def build_model(in_shape, out_shape, model='ann', args=None):
  if model=='manifold_mixup':
    model = ManifoldMixup(hidden_layers=[64,64,64,64], output_shape=out_shape)
  
  elif model=='jem':
    inputs = keras.Input(shape=(in_shape,))
    x = Dense(128, activation="sigmoid")(inputs)
    x = Dense(128, activation="sigmoid")(x)
    outputs = Dense(out_shape)(x)
    model = JEM.JEM(args.ld_lr, args.ld_std, args.ld_n, False, args.od_n, 
            args.od_lr, args.od_std, args.n_warmup, inputs=inputs, outputs=outputs)
  
  elif model=='jemo':
    inputs = keras.Input(shape=(in_shape,))
    x = Dense(128, activation="sigmoid")(inputs)
    x = Dense(128, activation="sigmoid")(x)
    outputs = Dense(out_shape)(x)
    model = JEM.JEM(args.ld_lr, args.ld_std, args.ld_n, True, args.od_n, 
            args.od_lr, args.od_std, args.n_warmup, inputs=inputs, outputs=outputs)
  
  elif model=='ann':
    model = tfk.Sequential([
        Dense(128, activation='relu', input_shape=(in_shape,)),
        Dense(128, activation='relu'),
        Dense(out_shape)
    ])
  else:
      raise AssertionError
  
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
