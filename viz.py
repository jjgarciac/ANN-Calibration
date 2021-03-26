import utils
import data_loader
import tensorflow as tf
import numpy as np
import argparse
import utils

def get_model_url(dataset, model):
    return ""

def load_args(url):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args_url = os.path.join(url, 'args.txt')
    if os.path.exists(args_url):
        with open(args_url, 'r') as f:
            ns = f.read()
            args = parser.parse_args(namespace=eval(ns))
    return args

def load_data(args):
    x_ood = None
    y_ood = None
    x_train, x_val, x_test, y_train, y_val, y_test, x_ood, y_ood = utils.prepare_ood(
        x_train, x_val, x_test, y_train, y_val, y_test, n_ood)
    return 0




