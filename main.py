import argparse
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as k
import matplotlib.pyplot as plt
import seaborn as sns
import io

from datetime import datetime
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import time

import data_loader, mixup
from models import build_model

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

def build_parser():
  parser = argparse.ArgumentParser(description='CLI Options',
                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--dataset", default="test",
                  help="name of dataset: abalone, arcene, arrhythmia, iris, phishing, moon, \
                          toy_Story, toy_Story_ood")
  parser.add_argument("--n_train", default=10000, type=int,
                  help="training data points for moon dataset")
  parser.add_argument("--n_test", default=1000, type=int,
                  help="testing data points for moon dataset")
  parser.add_argument("--train_noise", default=0.1, type=float,
                  help="noise for training samples")
  parser.add_argument("--test_noise", default=0.1, type=float,
                  help="noise for testing samples")
  

  parser.add_argument("--batch_size", default=16, type=int,
                  help="batch size used for training")
  parser.add_argument("--epochs", default=10, type=int,
                  help="number of epochs used for training")
  
  parser.add_argument("--mixup_scheme", default='random', type=str,
                  help="mix up strategy: random, knn, kfn, none")
  parser.add_argument("--out_of_class", default='false', type=str,
                  help="perform mixup across different classes")
  parser.add_argument("--n_neighbors", default=20, type=int,
                  help="number of neighbors to select from")
  parser.add_argument("--local_random", default='true', type=str,
          help="perform random mixup on only on batch samples")
  parser.add_argument("--alpha", default=0.4, type=float,
                  help="alpha for mixup")
  parser.add_argument("--manifold_mixup", action='store_true',
                  help="use manifold mixup instead of data mixup")
  
  parser.add_argument("--train_test_ratio", default=0.9, type=float,
                  help="split the dataset into training and testing")
  parser.add_argument("--shuffle", default='true', type=str,
                  help="shuffle after each epoch")
  parser.add_argument("--n_channels", default=1, type=int,
                  help="")

  parser.add_argument("--ood", action='store_true',
                  help="use ood samples if available on dataset.")
  return parser

def run():
  data = data_loader.load(DATASET,
                          n_train=N_TRAIN,
                          n_test=N_TEST,
                          train_noise=TRAIN_NOISE,
                          test_noise=TEST_NOISE,
                          ood=OOD)

  stratify = DATASET not in ["abalone"]
  
  if DATASET not in ['arcene', 'moon', 'toy_Story', 'toy_Story_ood']:
    x = data_loader.prepare_inputs(data['features'])
    y = data['labels']

    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        train_size=TRAIN_TEST_RATIO,
                                                        stratify=y if stratify else None)
    
  else:
    if DATASET == 'moon' or DATASET=='toy_Story' or DATASET=='toy_Story_ood':
      x_train, x_test = data['x_train'], data['x_val']
    else:
      x_train, x_test = data_loader.prepare_inputs(data['x_train'], data['x_val'])
    y_train, y_test = data['y_train'], data['y_val']

  
  # Generate validation split
  x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                  y_train,
                                                  train_size=TRAIN_TEST_RATIO,
                                                  stratify=y_train if stratify else None)

  x_train = x_train.astype(np.float32)
  x_val = x_val.astype(np.float32)
  x_test = x_test.astype(np.float32)
  
  print('Finish loading data')
  gdrive_rpath = './experiments'

  t = int(time.time())
  log_dir = os.path.join(gdrive_rpath, MODEL_NAME, '{}/logs'.format(t))
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
  file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')

  checkpoint_filepath = os.path.join(gdrive_rpath, MODEL_NAME, '{}/ckpt'.format(t))
  if not os.path.exists(checkpoint_filepath):
    os.makedirs(checkpoint_filepath)

  model_path= os.path.join(gdrive_rpath, MODEL_NAME, '{}/model'.format(format(t)))
  if not os.path.exists(model_path):
    os.makedirs(model_path)


  model_cp_callback = tf.keras.callbacks.ModelCheckpoint(
                                                        filepath=checkpoint_filepath,
                                                        save_weights_only=True,
                                                        monitor='val_accuracy',
                                                        mode='max',
                                                        save_best_only=True
                                                        )

  model = build_model(x_train.shape[1], y_train.shape[1], manifold_mixup=MANIFOLD_MIXUP)
  
  def plot_boundary(epoch, logs):
    # Use the model to predict the values from the validation dataset.
    xy = np.mgrid[-5:5:0.1, -5:5:0.1].reshape(2,-1).T
    hat_z = tf.nn.softmax(model(xy, training=False), axis=1)
    #scipy.special.softmax(hat_z, axis=1)
    c = np.sum(np.arange(hat_z.shape[1]+1)[1:]*hat_z, axis=1)
    #c = np.argmax(np.arange(6)[1:]*scipy.special.softmax(hat_z, axis=1), axis=1
    # xy = np.mgrid[-1:1.1:0.01, -2:2.1:0.01].reshape(2,-1).T
    figure = plt.figure(figsize=(8, 8))
    plt.scatter(xy[:,0], xy[:,1], c=c, cmap="brg")
    image = plot_to_image(figure) 
    # Log the confusion matrix as an image summary.
    with file_writer_cm.as_default():
      tf.summary.image("Boundaries", image, step=epoch)
  
  def plot_boundary_pretrain(epoch, logs):
    # Use the model to predict the values from the validation dataset.
    xy = np.mgrid[-1:1.1:0.01, -2:2.1:0.01].reshape(2,-1).T
    hat_z = tf.nn.softmax(model(xy, training=False), axis=1)
    #scipy.special.softmax(hat_z, axis=1)
    c = np.sum(np.arange(6)[1:]*hat_z, axis=1)
    #c = np.argmax(np.arange(6)[1:]*scipy.special.softmax(hat_z, axis=1), axis=1
    # xy = np.mgrid[-1:1.1:0.01, -2:2.1:0.01].reshape(2,-1).T
    figure = plt.figure(figsize=(8, 8))
    plt.scatter(xy[:,0], xy[:,1], c=c, cmap="brg")
    image = plot_to_image(figure) 
    # Log the confusion matrix as an image summary.
    with file_writer_cm.as_default():
      tf.summary.image("Boundaries_pretrain", image, step=epoch)


  
  if(DATASET=='toy_Story' or DATASET=='toy_Story_ood'): 
      border_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=plot_boundary)


  training_generator = mixup.data_generator(x_train, 
                                            y_train,
                                            batch_size=BATCH_SIZE, 
                                            n_channels=N_CHANNELS,
                                            shuffle=SHUFFLE,
                                            mixup_scheme=MIXUP_SCHEME,
                                            k=N_NEIGHBORS,
                                            alpha=ALPHA,
                                            local=LOCAL_RANDOM,
                                            out_of_class=OUT_OF_CLASS,
                                            manifold_mixup=MANIFOLD_MIXUP
                                            )
  
  validation_generator = mixup.data_generator(x_val, 
                                              y_val,
                                              batch_size=x_val.shape[0], 
                                              n_channels=N_CHANNELS,
                                              shuffle=False,
                                              mixup_scheme='none',
                                              alpha=0,
                                              manifold_mixup=MANIFOLD_MIXUP
                                              )
  
  test_generator = mixup.data_generator(x_test, 
                                        y_test,
                                        batch_size=x_test.shape[0], 
                                        n_channels=N_CHANNELS,
                                        shuffle=False,
                                        mixup_scheme='none',
                                        alpha=0,
                                        manifold_mixup=MANIFOLD_MIXUP
                                        )

  # Pretraining
  # if DATASET=='toy_Story': 
  #   pre_x = np.mgrid[-1:1.1:0.01, -2:2.1:0.01].reshape(2,-1).T 
  #   pre_y = .2*np.ones(shape=[pre_x.shape[0], 5])
  #   model.fit(x=pre_x, y=pre_y, epochs=1, callbacks=[border_callback_pretrain])

  training_history = model.fit(x=training_generator, 
                                validation_data=validation_generator,
                                epochs=EPOCHS, 
                                callbacks=[
                                           tensorboard_callback,
                                           model_cp_callback,
                                           border_callback
                                           ],
                                )
  print(model.summary())
  #model.load_weights(checkpoint_filepath)
  #model.save(model_path)
  print('Tensorboard callback directory: {}'.format(log_dir))
  
  metric_file = os.path.join(gdrive_rpath, MODEL_NAME, '{}/results.txt'.format(t))
  loss = model.evaluate(validation_generator, return_dict=True)
  
  with open(metric_file, "w") as f:
    f.write(str(loss))

if __name__ == "__main__":
  parser = build_parser()

  args = parser.parse_args()

  BATCH_SIZE = args.batch_size
  EPOCHS = args.epochs
  DATASET = args.dataset
  N_TRAIN = args.n_train
  N_TEST = args.n_test
  TRAIN_NOISE = args.train_noise
  TEST_NOISE = args.test_noise
  TRAIN_TEST_RATIO = args.train_test_ratio
  MANIFOLD_MIXUP = args.manifold_mixup
  OOD = args.ood
  MIXUP_SCHEME = args.mixup_scheme
  if MIXUP_SCHEME == 'random':
    N_NEIGHBORS = 0
  else:
    N_NEIGHBORS = args.n_neighbors

  OUT_OF_CLASS = True if args.out_of_class == 'true' else False
  LOCAL_RANDOM = True if args.local_random == 'true' else False
  SHUFFLE = True if args.shuffle == 'true' else False
  N_CHANNELS = args.n_channels
  if MIXUP_SCHEME != 'none':
    ALPHA = args.alpha
  else:
    ALPHA = 0

  MODEL_NAME = '{}/r{}-b{}-e{}-a{}-{}-n{}-l{}-o{}{}'.format(DATASET,
                                                            TRAIN_TEST_RATIO,
                                                            BATCH_SIZE,
                                                            EPOCHS,
                                                            ALPHA,
                                                            MIXUP_SCHEME,
                                                            N_NEIGHBORS,
                                                            1 if LOCAL_RANDOM else 0,
                                                            1 if OUT_OF_CLASS else 0,
                                                            "-manifold" if MANIFOLD_MIXUP else ""
                                                            )
  
  
  run()