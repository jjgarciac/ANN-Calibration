import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.layers import Dense
import math
from metrics import *
from model import *
import keras #Try and remove this import
from model import JEM
from model import JEHM

layers = keras.layers

def gauss_pdf(x, name=None):
    return (1/tf.sqrt(2*math.pi))*tf.math.exp((-tf.pow(x, 2))/2)

def build_model(in_shape, out_shape, model='ann', args=None):
  model_img = args.dataset in {'mnist','cifar10','kmnist', 'svhn'}
  inputs = keras.Input(shape=in_shape)
  if model_img:
    cx = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    cx = layers.MaxPooling2D((2, 2))(cx)
    cx = layers.Conv2D(64, (3, 3), activation='relu')(cx)
    cx = layers.MaxPooling2D((2, 2))(cx)
    cx = layers.Conv2D(64, (3, 3), activation='relu')(cx)
    cx = layers.Flatten()(cx)
    cx = layers.Dense(64, activation='relu')(cx)
    cnn_out = layers.Dense(args.cnn_out_size)(cx)
    #TODO: Fix this ugly change of variables
    model_inputs = cnn_out
  else:
    model_inputs = inputs
  

  if model=='manifold_mixup':
    model = ManifoldMixup(hidden_layers=[64,64,64,64], output_shape=out_shape)
  
  elif model=='jem':
    x = Dense(128, activation="sigmoid")(model_inputs)
    x = Dense(128, activation="sigmoid")(x)
    outputs = Dense(out_shape)(x)
    #z = Dense(out_shape)(x)
    #p_x = tf.reduce_logsumexp(z, axis=1, keepdims=True)
    #outputs = p_x*z
    model = JEM.JEM(args.ld_lr, args.ld_std, args.ld_n, False, args.od_n, 
            args.od_lr, args.od_std, args.od_l, args.n_warmup, inputs=inputs, outputs=outputs)
  
  elif model=='jehm':
    x = Dense(128, activation="sigmoid")(model_inputs)
    x = Dense(128, activation="sigmoid")(x)
    outputs = Dense(out_shape)(x)
    model = JEHM.JEHM(args.ld_lr, args.ld_std, args.ld_n, False, args.od_n, 
            args.od_lr, args.od_std, args.od_l, args.n_warmup, inputs=inputs, outputs=outputs)
  
  elif model=='jehmo':
    x = Dense(128, activation="sigmoid")(model_inputs)
    x = Dense(128, activation="sigmoid")(x)
    outputs = Dense(out_shape)(x)
    model = JEHM.JEHM(args.ld_lr, args.ld_std, args.ld_n, True, args.od_n, 
            args.od_lr, args.od_std, args.od_l, args.n_warmup, inputs=inputs, outputs=outputs)
  
  elif model=='jemo':
    x = Dense(128, activation="sigmoid")(model_inputs)
    x = Dense(128, activation="sigmoid")(x)
    outputs = Dense(out_shape)(x)
    model = JEM.JEM(args.ld_lr, args.ld_std, args.ld_n, True, args.od_n, 
            args.od_lr, args.od_std, args.od_l, args.n_warmup, inputs=inputs, outputs=outputs)
  
  elif model=='ann':
    x = Dense(128, activation='relu')(model_inputs)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(out_shape)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
  else:
      raise AssertionError
  
  optimizer = tfk.optimizers.Adam()
  loss = tfk.losses.CategoricalCrossentropy(from_logits=True)

  metrics = [ECE_metrics(name='ECE', num_of_bins=10),
             OE_metrics(name='OE', num_of_bins=10),
             tfk.metrics.CategoricalAccuracy('accuracy', dtype=tf.float32),
             AUC_of_OOD(name='auc_ood')]
    
  model.compile(optimizer=optimizer, 
                loss=loss, 
                metrics=metrics,
                run_eagerly=True)
  return model
