import utils
import data_loader
import tensorflow as tf
import numpy as np
import argparse
import utils
import pandas as pd
import models
import matplotlib.pyplot as plt
import os
from argparse import Namespace
from sklearn.model_selection import train_test_split

def isfloat(x):
    try:
        a = float(x)
    except (TypeError, ValueError):
        return False
    else:
        return True

def isint(x):
    try:
        a = float(x)
        b = int(a)
    except (TypeError, ValueError):
        return False
    else:
        return a == b

def get_model_url(model, dataset, id):
    return os.path.join('./experiments', *[model, dataset, id])

def old_load_args(url):
    parser = argparse.ArgumentParser()
    #parser.add_argument('--a', default=1)
    #args = parser.parse_args()
    args=None
    args_url = os.path.join(url, 'args.txt')
    if os.path.exists(args_url):
        with open(args_url, 'r') as f:
            ns = f.read()
            args = parser.parse_args(namespace=eval(ns))
    return args

def load_args(url):
    parser = argparse.ArgumentParser()
    args={}
    args_url = os.path.join(url, 'args.txt')
    if os.path.exists(args_url):
        with open(args_url, 'r') as f:
            ns = f.read()
            for arg in ns[10:].split(','):
                arg = arg.split('=')
                arg[1] = arg[1].strip('\'')
                arg[1] = arg[1].rstrip(')')
                v = arg[1]
                if(arg[1]=='True'):
                    v=True
                if(arg[1]=='False'):
                    v=False
                if(isfloat(arg[1])):
                    v=float(arg[1])
                if(isint(arg[1])):
                    v=int(arg[1])
                args[arg[0].strip()]=v
    return Namespace(**args)

def load_data(args):
    data = data_loader.load(args.dataset,
                          n_train=args.n_train,
                          n_test=args.n_test,
                          train_noise=args.train_noise,
                          test_noise=args.test_noise)
    stratify = args.dataset not in ["abalone", "segment"]
    if args.dataset not in ['arcene', 'moon', 'toy_Story', 'toy_Story_ood', 'segment']:
        print(args.dataset)
        x = data_loader.prepare_inputs(data['features'])
        y = data['labels']
        x_train, x_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            train_size=args.train_test_ratio,
                                                        stratify=y if stratify else None)
    else:
        if args.dataset == 'moon' or args.dataset=='toy_Story' or \
           args.dataset=='toy_Story_ood':
            x_train, x_test = data['x_train'], data['x_val']
        else:
            x_train, x_test = data_loader.prepare_inputs(data['x_train'], data['x_val'])
        y_train, y_test = data['y_train'], data['y_val']
  # Generate validation split
    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                  y_train,
                                                  train_size=args.train_test_ratio,
                                                  stratify=y_train if stratify else None)
    x_train = x_train.astype(np.float32)
    x_val = x_val.astype(np.float32)
    x_test = x_test.astype(np.float32)
    
    n_mean = np.mean(x_train, axis=0)
    n_std = np.var(x_train, axis=0)**.5 
  
    x_train = (x_train-n_mean)/n_std
    x_val = (x_val-n_mean)/n_std
    x_test = (x_test-n_mean)/n_std

    try:
        if args.n_ood>0 and y_val.shape[1]>args.n_ood:
            n_ood = y_val.shape[1]-args.n_ood-1
            return utils.prepare_ood(x_train, x_val, x_test, y_train, y_val, y_test, n_ood, args.norm)
    except AttributeError:
        #print(x_train, x_val, x_test, y_train, y_val, y_test)
        return x_train, x_val, x_test, y_train, y_val, y_test, 0, 0
    return x_train, x_val, x_test, y_train, y_val, y_test, 0, 0

def load_model(url, in_shape, out_shape, args):
    checkpoint_filepath = os.path.join(url, 'ckpt')
    model = models.build_model(in_shape, out_shape, args.model, args)
    model.load_weights(checkpoint_filepath)
    return model

def leave_cvx_hull(model_list, x, y):
    fig, axs = plt.subplots(1, 3)
    fig.suptitle('Leaving cvx hull')
    for i, y_lbl in enumerate(['accuracy', 'confidence', 'entropy']):
        for model in model_list:
            y_list = []
            for u in range(0, 100):
                y_plot=0
                t = np.random.uniform(size=x.shape)
                py_x = tf.nn.softmax(model(x + u*t))
                if y_lbl=='entropy':
                    y_plot = -tf.reduce_mean(tf.reduce_sum(py_x*tf.math.log(py_x), axis=1))
                if y_lbl=='accuracy':
                    y_plot = tf.reduce_mean(tf.cast(
                        tf.argmax(py_x, axis=1)==tf.argmax(y, axis=1), tf.float32))
                if y_lbl=='confidence':
                    y_plot = tf.reduce_mean(tf.reduce_max(py_x, 1))
                y_list.append(y_plot.numpy())
            axs[i].plot(np.array(y_list), label=model.name)
        axs[i].set(xlabel='perturbation', ylabel=y_lbl+' (mean)')
        axs[i].legend()

def confidence_plot(model, x, xo):
    p_in = tf.max(tf.nn.softmax(model(x)), axis=1)
    p_out = tf.max(tf.nn.softmax(model(xo)), axis=1)
    plt.ylabel('Frequency')
    plt.xlabel('Confidence')
    plt.xlim([0, 1])
    plt.hist(p_in, bins=20, color='blue', label='In', alpha=.5)
    plt.hist(p_out, bins=20, color='red', label='Out', alpha=.5)
    plt.legend()
    return 0

def calibration_plot(model, x, y, ece):
    py_x = tf.nn.softmax(model(x))
    p = tf.max(py_x, axis=1)
    
    hat_y = tf.argmax(py_x, axis=1)
    y = tf.argmax(y, axis=1)
    acc = tf.cast(hat_y==y, tf.float32)
 
    idx = tf.argsort(p)
    p = p[idx]
    acc = acc[idx]

    plt.title(f'Calibration {model.name}: {ece}')
    plt.ylabel('Frequency')
    plt.xlabel('ACC/Conf')
    plt.xlim([0, 1])
    plt.hist(acc, bins=20, color='blue', label='accuracy', alpha=.5)
    plt.hist(p, bins=20, color='red', label='confidence', alpha=.5)
    plt.legend()
    return 0

def analyze_features(model, x, xo, idx):
    plt.hist(x[:, idx], bins=20, color='blue', label='sample', alpha=.5)
    plt.hist(xo[:, idx], bins=20, color='red', label='ood', alpha=.5)
    if model.name in ['jemo', 'jehmo']:
        xgo = model.sample_ood(x)
        plt.hist(xgo[:, idx], bins=20, color='green', label='gen_o', alpha=.5)
    plt.legend()
    return 0
