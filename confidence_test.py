import argparse
import os
import numpy as np
import tensorflow.keras as k
import matplotlib.pyplot as plt
import io
import callbacks as cb
from sklearn.model_selection import train_test_split
import time

import data_loader, mixup
from models import build_model
import tensorflow as tf
tf.executing_eagerly()

def build_parser():
  parser = argparse.ArgumentParser(description='cli options',
              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  # dataset parameters
  parser.add_argument("--dataset", default="segment",
                  help="name of dataset: abalone, arcene, arrhythmia, iris, \
                          phishing, moon, sensorless_drive, segment,\
                          htru2, heart disease, mushroom, wine, \
                          toy_story, toy_story_ood")
  parser.add_argument("--n_train", default=10000, type=int,
                  help="training data points for moon dataset")
  parser.add_argument("--n_test", default=1000, type=int,
                  help="testing data points for moon dataset")
  parser.add_argument("--train_noise", default=0.1, type=float,
                  help="noise for training samples")
  parser.add_argument("--test_noise", default=0.1, type=float,
                  help="noise for testing samples")
  
  # training dataset parameters
  parser.add_argument("--batch_size", default=16, type=int,
                  help="batch size used for training")
  parser.add_argument("--epochs", default=10, type=int,
                  help="number of epochs used for training")
  parser.add_argument("--shuffle", default='true', type=str,
                  help="shuffle after each epoch")
  parser.add_argument("--monitor", default='val_accuracy', type=str,
                  help="metric to monitor")
  parser.add_argument("--ood", action='store_true',
                  help="use ood samples if available on dataset.")
  
  # model parameters
  parser.add_argument("--model", default='ann', type=str,
                  help="available models: ann, jem, jemo, manifold_mixup")

  # mixup scheme setup
  parser.add_argument("--mixup_scheme", default='none', type=str,
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
  parser.add_argument("--n_channels", default=1, type=int,
                  help="")
  
  # jem model parameters
  parser.add_argument("--jem", action='store_true',
                  help="flag to use jem model")
  parser.add_argument("--ld_lr", default=.2, type=float,
                  help="gradient step scale for jem p(x) sampler")
  parser.add_argument("--ld_std", default=1e-2, type=float,
                  help="sampling noise std for jem")
  parser.add_argument("--ld_n", default=20, type=float,
                  help=" for jem ood loss")
  parser.add_argument("--od_n", default=25, type=int,
                  help="number of ood points to sample from jem")
  parser.add_argument("--od_lr", default=.2, type=float,
                  help="gradient scale for jem ood samples.")
  parser.add_argument("--od_std", default=.1, type=float,
                  help="sampling noise for jem ood samples.")
  parser.add_argument("--od_l", default=.01, type=float,
                  help="jem ood loss scale.")
  parser.add_argument("--n_warmup", default=50, type=int,
                  help="training steps before introducing ood points in jem.")
  # image log prameters 
  parser.add_argument("--n_ood", default=0, type=int,
                  help="number of classes to separate from dataset.")
  return parser

def run():
  data = data_loader.load(DATASET,
                          n_train=N_TRAIN,
                          n_test=N_TEST,
                          train_noise=TRAIN_NOISE,
                          test_noise=TEST_NOISE)

  STRATIFY = DATASET not in ["abalone", "segment"]
  
  if DATASET not in ['arcene', 'moon', 'toy_story', 'toy_story_ood', 'segment']:
    x = data_loader.prepare_inputs(data['features'])
    y = data['labels']
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        train_size=TRAIN_TEST_RATIO,
                                                        stratify=y if STRATIFY else none)
  else:
    if DATASET == 'moon' or DATASET=='toy_story' or DATASET=='toy_story_ood':
      x_train, x_test = data['x_train'], data['x_val']
    else:
      x_train, x_test = data_loader.prepare_inputs(data['x_train'], data['x_val'])
    y_train, y_test = data['y_train'], data['y_val']

  # generate validation split
  x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                  y_train,
                                                  train_size=TRAIN_TEST_RATIO,
                                                  stratify=y_train if STRATIFY  else None)

  x_train = x_train.astype(np.float32)
  x_val = x_val.astype(np.float32)
  x_test = x_test.astype(np.float32)

  # delete for categorical datasets
  n_mean = np.mean(x_train, axis=0)
  n_std = np.var(x_train, axis=0)**.5
  
  x_train = (x_train-n_mean)/n_std
  x_val = (x_val-n_mean)/n_std
  x_test = (x_test-n_mean)/n_std
  
  n_classes = y_val.shape[1]
  if N_OOD>0 and n_classes>N_OOD:
      n_ood = n_classes-N_OOD-1
      idx_train_ood = np.argmax(y_train, axis=1)>n_ood
      idx_train_in = np.argmax(y_train, axis=1)<=n_ood
      idx_test_ood = np.argmax(y_test, axis=1)>n_ood
      idx_test_in = np.argmax(y_test, axis=1)<=n_ood
      idx_val_ood = np.argmax(y_val, axis=1)>n_ood
      idx_val_in = np.argmax(y_val, axis=1)<=n_ood
      
      x_test_ood = x_test[idx_test_ood]
      y_test_ood = y_test[idx_test_ood][n_ood+1:]
      x_train_ood = x_train[idx_train_ood]
      y_train_ood = y_train[idx_train_ood][n_ood+1:]
      x_val_ood = x_val[idx_val_ood]
      y_val_ood = y_val[idx_val_ood][n_ood+1:]
      
      x_train = x_train[idx_train_in]
      x_test = x_test[idx_test_in]
      x_val = x_val[idx_val_in]
      y_train = y_train[idx_train_in][:n_ood+1]
      y_test = y_test[idx_test_in][:n_ood+1]
      y_val = y_val[idx_val_in][:n_ood+1]
  
  print('Finish loading data')
  gdrive_rpath = './experiments'

  t = int(time.time())
  log_dir = os.path.join(gdrive_rpath, MODEL_NAME, '{}'.format(t))
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
  file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')

  checkpoint_filepath = os.path.join(log_dir, 'ckpt')
  if not os.path.exists(checkpoint_filepath):
    os.makedirs(checkpoint_filepath)

  model_path= os.path.join(log_dir, 'model')
  if not os.path.exists(model_path):
    os.makedirs(model_path)

  model_cp_callback = tf.keras.callbacks.ModelCheckpoint(
                                                        filepath=checkpoint_filepath,
                                                        save_weights_only=True,
                                                        monitor=MONITOR,
                                                        mode='max',
                                                        save_best_only=True)

  model = build_model(x_train.shape[1], y_train.shape[1], MODEL, args)

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
                                            manifold_mixup=MANIFOLD_MIXUP)
  
  validation_generator = mixup.data_generator(x_val, 
                                              y_val,
                                              batch_size=x_val.shape[0], 
                                              n_channels=N_CHANNELS,
                                              shuffle=False,
                                              mixup_scheme='none',
                                              alpha=0,
                                              manifold_mixup=MANIFOLD_MIXUP)
  
  test_generator = mixup.data_generator(x_test, 
                                        y_test,
                                        batch_size=x_test.shape[0], 
                                        n_channels=N_CHANNELS,
                                        shuffle=False,
                                        mixup_scheme='none',
                                        alpha=0,
                                        manifold_mixup=MANIFOLD_MIXUP)

  callbacks=[tensorboard_callback, model_cp_callback]
  if DATASET=='Toy_story' or DATASET=='Toy_story_ood':
    border_callback = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=cb.plot_boundary)
    callbacks+=[border_callback]
  if MODEL=='jem':
    callbacks+=[cb.jem_n_epochs()]

  training_history = model.fit(x=training_generator, 
                                validation_data=validation_generator,
                                epochs=EPOCHS, 
                                callbacks=callbacks)

  model.load_weights(checkpoint_filepath)
  #model.save(model_path)
  print('Tensorboard callback directory: {}'.format(log_dir))
  
  metric_file = os.path.join(gdrive_rpath, 'results.txt')
  loss = model.evaluate(test_generator, return_dict=True)
  
  z_in = tf.nn.softmax(model(np.concatenate([x_test, x_val], axis=0)))
  c_in = tf.math.reduce_max(z_in, axis=-1)
  acc_in = tf.reduce_mean(tf.cast(tf.math.argmax(z_in, axis=-1)==y_test, tf.float32))
  
  z_out = tf.nn.softmax(model(np.concatenate([x_train_ood, x_test_ood, x_val_ood], axis=0)))
  c_out = tf.math.reduce_max(z_out, axis=-1)
  
  z_train = tf.nn.softmax(model(x_train))
  c_train = tf.math.reduce_max(z_train, axis=-1)
  
  # Plot histogram from confidences
  plt.hist(c_in, bins=20, color='blue', label='In')
  plt.hist(c_out, bins=20, color='red', label='Out')
  #plt.hist(c_train, density=True, bins=20, color='green', label='Train_in')
  plt.ylabel('Frequency')
  plt.xlabel('Confidence')
  plt.xlim([0, 1])
  plt.legend()
  plt.savefig(os.path.join(log_dir, 'confidence.png'), dpi=300)
  plt.close()
  plt.hist(c_in, density=True, bins=20, color='blue', label='Confidence')
  plt.hist(acc_in, density=True, bins=20, color='red', label='Accuracy')
  plt.ylabel('Fraction')
  plt.xlabel('Confidence')
  plt.xlim([0, 1])
  plt.legend()
  plt.savefig(os.path.join(log_dir, 'acc_conf.png'), dpi=300)
  plt.close()
  with open(metric_file, "a+") as f:
      f.write(f"{MODEL}, {DATASET}, {t}, {loss['accuracy']:.3f}," \
              f"{loss['ece_metrics']:.3f}, {loss['oe_metrics']:.3f}," \
              f"{loss['loss']:.3f}\n")
  
  arg_file = os.path.join(log_dir, 'args.txt')
  with open(arg_file, "w+") as f:
    f.write(str(args))

if __name__ == "__main__":
  parser = build_parser()

  args = parser.parse_args()
  MODEL = args.model
  BATCH_SIZE = args.batch_size
  EPOCHS = args.epochs
  DATASET = args.dataset
  N_TRAIN = args.n_train
  N_TEST = args.n_test
  TRAIN_NOISE = args.train_noise
  TEST_NOISE = args.test_noise
  TRAIN_TEST_RATIO = args.train_test_ratio
  MANIFOLD_MIXUP = args.manifold_mixup
  HYBRID_MODEL = args.jem
  MONITOR = args.monitor
  OOD = args.ood
  N_OOD = args.n_ood
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

  MODEL_NAME = '{}/{}'.format(MODEL, DATASET)
  run()
