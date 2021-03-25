import tensorflow as tf
import keras

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def plot_boundary(epoch, logs):
    """Callback to plot toy_story decision boundaries"""
    xy = np.mgrid[-5:5:0.1, -5:5:0.1].reshape(2,-1)
    hat_z = tf.nn.softmax(self.model(xy, training=False), axis=1)
    c = np.sum(np.arange(hat_z.shape[1]+1)[1:]*hat_z, axis=1)
    figure = plt.figure(figsize=(8, 8))
    plt.scatter(xy[:,0], xy[:,1], c=c, cmap="brg")
    image = plot_to_image(figure) 
    with file_writer_cm.as_default():
        tf.summary.image("boundaries", image, step=epoch)

#def jem_n_epochs(epoch, logs):
#    """Only valid for JEM model.
#    Helper to update JEM's n_epoch parameter. This in turn controls
#    when ood samples will be used for training."""
#    self.model.n_epochs+=1

class jem_n_epochs(keras.callbacks.Callback):
    def __init__(self):
        super(jem_n_epochs, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        self.model.n_epochs+=1
