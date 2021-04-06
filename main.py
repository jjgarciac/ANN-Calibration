import argparse
import os
import numpy as np
import tensorflow.keras as k
import matplotlib.pyplot as plt
import io
import callbacks as cb
from sklearn.model_selection import train_test_split
import time
import utils
import data_loader, mixup
from models import build_model
import tensorflow as tf
tf.executing_eagerly()

def build_parser():
  parser = argparse.ArgumentParser(description='CLI Options',
                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  # Dataset parameters
  parser.add_argument("--dataset", default="segment",
                  help="name of dataset: abalone, arcene, arrhythmia, iris, \
                          phishing, moon, sensorless_drive, segment,\
                          htru2, heart disease, mushroom, wine, \
                          toy_Story, toy_Story_ood")
  parser.add_argument("--n_train", default=10000, type=int,
                  help="training data points for moon dataset")
  parser.add_argument("--n_test", default=1000, type=int,
                  help="testing data points for moon dataset")
  parser.add_argument("--train_noise", default=0.1, type=float,
                  help="noise for training samples")
  parser.add_argument("--test_noise", default=0.1, type=float,
                  help="noise for testing samples")
  parser.add_argument("--n_ood", default=0, type=int,
                  help="number of classes to separate from dataset.")
  parser.add_argument("--norm", action="store_true",
                  help="Normalize dataset")
  
  # Training Dataset parameters
  parser.add_argument("--batch_size", default=16, type=int,
                  help="batch size used for training")
  parser.add_argument("--epochs", default=10, type=int,
                  help="number of epochs used for training")
  parser.add_argument("--shuffle", default='true', type=str,
                  help="shuffle after each epoch")
  parser.add_argument("--monitor", default='val_accuracy', type=str,
                  help="Metric to monitor")
  parser.add_argument("--ood", action='store_true',
                  help="use ood samples if available on dataset.")
  
  # Model parameters
  parser.add_argument("--model", default='ann', type=str,
                  help="Available models: ann, jem, jehm, \
                      jemo, jehmo, manifold_mixup")

  # Mixup scheme setup
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
  
  # JEM model parameters
  parser.add_argument("--JEM", action='store_true',
                  help="Flag to use JEM model")
  parser.add_argument("--ld_lr", default=.2, type=float,
                  help="Gradient step scale for JEM p(x) sampler")
  parser.add_argument("--ld_std", default=1e-2, type=float,
                  help="Sampling noise std for JEM")
  parser.add_argument("--ld_n", default=20, type=int,
                  help=" for JEM ood loss")
  parser.add_argument("--od_n", default=25, type=int,
                  help="Number of ood points to sample from JEM")
  parser.add_argument("--od_lr", default=.2, type=float,
                  help="Gradient scale for JEM ood samples.")
  parser.add_argument("--od_std", default=.1, type=float,
                  help="Sampling noise for JEM ood samples.")
  parser.add_argument("--od_l", default=.01, type=float,
                  help="JEM ood loss scale.")
  parser.add_argument("--n_warmup", default=50, type=int,
                  help="Training steps before introducing ood points in JEM.")
  return parser

def run():
  data = data_loader.load(DATASET,
                          n_train=N_TRAIN,
                          n_test=N_TEST,
                          train_noise=TRAIN_NOISE,
                          test_noise=TEST_NOISE)

  stratify = DATASET not in ["abalone", "segment"]
  
  if DATASET not in ['arcene', 'moon', 'toy_Story', 'toy_Story_ood', 'segment']:
    print(DATASET)
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
  
  if NORM:
    print("Normalizing dataset")
    n_mean = np.mean(x_train, axis=0)
    n_std = np.var(x_train, axis=0)**.5 
      
    x_train = (x_train-n_mean)/n_std
    x_val = (x_val-n_mean)/n_std
    x_test = (x_test-n_mean)/n_std
  
  if N_OOD>0 and y_val.shape[1]>N_OOD:
    n_ood = y_val.shape[1]-N_OOD-1
    x_train, x_val, x_test, y_train, y_val, y_test, x_ood, y_ood = utils.prepare_ood(
        x_train, x_val, x_test, y_train, y_val, y_test, n_ood)

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
                                                        save_best_only=True,
                                                        verbose=1)

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
  if MODEL in ['jem', 'jemo', 'jehm', 'jehmo']:
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
   
  with open(metric_file, "a+") as f:
      f.write(f"{MODEL}, {DATASET}, {t}, {loss['accuracy']:.3f}," \
              f"{loss['ece_metrics']:.3f}, {loss['oe_metrics']:.3f}," \
              f"{loss['loss']:.3f}, {N_OOD}\n")
  
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
  HYBRID_MODEL = args.JEM
  MONITOR = args.monitor
  OOD = args.ood
  N_OOD = args.n_ood
  NORM = args.norm
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
