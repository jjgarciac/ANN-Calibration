import tensorflow as tf
import data_loader
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_rgba
import numpy as np

def scatter(x, y, color, alphas, **kwarg):
    #r, g, b = to_rgb(color)
    # r, g, b, _ = to_rgba(color)
    # Color map I want to use
    cm = plt.cm.get_cmap('brg')
    # Get the colormap colors for my data
    my_cmap = cm(plt.Normalize(color.min(), color.max())(color))
    # Set alpha
    my_cmap[:, -1] = alphas
    # Create new colormap
    #my_cmap = ListedColormap(my_cmap)
    #color = [(r, g, b, alpha) for alpha in alpha_arr]
    plt.scatter(x, y, color=my_cmap, edgecolors='black', **kwarg)

X, Y = data_loader.make_toy_Story(45)
Y_oh = tf.one_hot(Y, len(np.unique(Y)), dtype=tf.float32, off_value=0.0001).numpy()
X_tmp = np.zeros_like(X)
Y_oh_tmp = np.zeros_like(Y_oh)
for i in range(2):
    p = np.random.permutation(len(X))
    X_tmp = .7*X_tmp + .3*X[p]
    Y_oh_tmp = .7*Y_oh_tmp + .3*Y_oh[p]
    X = np.concatenate([X, X_tmp], axis=0)
    Y_oh = np.concatenate([Y_oh, Y_oh_tmp], axis=0)
#X = np.concatenate([X, X_tmp])
#Y_oh = np.concatenate([Y_oh, Y_oh_tmp])
alpha = 1-(np.sum(np.log(Y_oh)*Y_oh, axis=1)/np.log(.2))
c = np.sum(np.arange(Y_oh.shape[1]+1)[1:]*Y_oh, axis=1)
scatter(X[:, 0], X[:, 1], c, alpha)
plt.show()
