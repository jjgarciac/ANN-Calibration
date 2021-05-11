import utils
import torch as t, torch.nn as nn, torch.nn.functional as tnnF, torch.distributions as tdist
from torch.utils.data import DataLoader, Dataset
import torchvision as tv, torchvision.transforms as tr
import os
import sys
import argparse
#import ipdb
import numpy as np
import json
# Sampling
from tqdm import tqdm
import tensorflow as tf
import torch as t

def init_random(bs, feat_size):
    return tf.random.uniform((bs, feat_size), minval=-1, maxval=1)


def get_buffer(buffer_size, feat_size, x=None):
    replay_buffer = init_random(buffer_size, feat_size)
    if not x is None:
        x_inds = np.random.randint(0, buffer_size, (x.shape[0],))  # t.randint(0, buffer_size, (bs,))
        replay_buffer = tf.tensor_scatter_nd_update(replay_buffer,
                                                    x_inds[:, None],
                                                    x)
    return replay_buffer

