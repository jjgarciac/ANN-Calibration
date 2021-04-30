from sklearn.neighbors import NearestNeighbors
import tensorflow.keras as keras
import numpy as np


def mixup_uncertainty(batch_x, batch_y, x, y, alpha=1.0, out_of_class=False,
        local=True, manifold_mixup=False):
    
    batch_size = batch_x.shape[0]
    
    if alpha > 0:
        lam = np.random.beta(alpha, alpha, batch_size)
    else:
        lam = np.ones(batch_size)
    rlam = lam.reshape((batch_size, 1))
    
    if local:
      index = np.random.permutation(batch_size)
      mixed_x = rlam * batch_x + (1 - rlam) * batch_x[index, :]
      mixed_y = rlam * batch_y + (1 - rlam) * batch_y[index, :]
      input_x  = [batch_x, batch_x[index, :], rlam] if manifold_mixup else mixed_x
    else:
      index = np.random.permutation(x.shape[0])[:batch_size]
      mixed_x = rlam * batch_x + (1 - rlam) * x[index, :]
      mixed_y = rlam * batch_y + (1 - rlam) * y[index, :]
      input_x  = [batch_x, x[index, :], rlam] if manifold_mixup else mixed_x

    return input_x, mixed_y

def mixup_random(batch_x, batch_y, x, y, alpha=1.0, out_of_class=False,
                 local=True, manifold_mixup=False):
    batch_size = batch_x.shape[0]

    if alpha > 0:
        lam = np.random.beta(alpha, alpha, batch_size)
    else:
        lam = np.ones(batch_size)
    #TODO: Fix this logic
    rlamy = lam.reshape((batch_size, 1))
    if len(batch_x.shape)>2:
        rlamx = lam.reshape((batch_size, 1, 1, 1))
    else:
        rlamx = rlamy
    
    if local:
      index = np.random.permutation(batch_size)
      mixed_x = rlamx * batch_x + (1 - rlamx) * batch_x[index, :]
      mixed_y = rlamy * batch_y + (1 - rlamy) * batch_y[index, :]
      input_x  = [batch_x, batch_x[index, :], rlamx] if manifold_mixup else mixed_x
    else:
      index = np.random.permutation(x.shape[0])[:batch_size]
      mixed_x = rlamx * batch_x + (1 - rlamx) * x[index, :]
      mixed_y = rlamy * batch_y + (1 - rlamy) * y[index, :]
      input_x  = [batch_x, x[index, :], rlamx] if manifold_mixup else mixed_x

    return input_x, mixed_y


def knn(x, k=None):
    """Find k-nearest neighbor of each sample"""
    
    if k is None:
      k = x.shape[0]
    neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(x)
    return neigh

def mixup_knn(batch_x, batch_y, x, y, neigh, k, alpha=1.0, nearest=True,
              out_of_class=False, manifold_mixup=False):
    batch_size = batch_x.shape[0]

    if alpha > 0:
        lam = np.random.beta(alpha, alpha, batch_size)
    else:
        lam = np.ones(batch_size)
    rlam = lam.reshape((batch_size, 1))
        
    indices = neigh.kneighbors(batch_x, return_distance=False)

    if out_of_class:
      selected_indices = []
      for x_indices, y_label in zip(indices, batch_y):
        n = 0
        if not nearest:
          x_indices = x_indices[::-1]
        tmp = []
        for i in x_indices:
          if not np.equal(y[i], y_label).all():
            tmp.append(i)
            n += 1
            if n >= k:
                break
            
        selected_indices.append(tmp)
      selected_indices = np.array(selected_indices)
    else:
      if nearest:
        selected_indices = indices[:,:k]
      else:
        selected_indices = indices[:, -1*k:]

    indices_t = selected_indices.T
    np.random.shuffle(indices_t)
    shuffle_indices = indices_t.T
    index = shuffle_indices[:, 0]
    
    mixed_x = rlam * batch_x + (1 - rlam) * x[index, :]
    mixed_y = rlam * batch_y + (1 - rlam) * y[index, :]

    input_x  = [batch_x, x[index, :], rlam] if manifold_mixup else mixed_x

    return input_x, mixed_y


def mixup(batch_x, batch_y, x, y, k, scheme, alpha, local, out_of_class, manifold_mixup, neigh):
  if scheme in ['random', 'none']:
    return mixup_random(batch_x, batch_y, x, y, alpha=alpha, out_of_class=out_of_class,
                        local=local, manifold_mixup=manifold_mixup)
  nearest = True if scheme == 'knn' else False
  return mixup_knn(batch_x, batch_y, x, y, neigh, k=k, alpha=alpha,
                   nearest=nearest,out_of_class=out_of_class, manifold_mixup=manifold_mixup)

class data_generator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, 
                 x, 
                 y, 
                 batch_size=32, 
                 n_channels=1,
                 shuffle=True,
                 mixup_scheme='random',
                 k=None,
                 alpha=1.0,
                 local=False,
                 out_of_class=False,
                 manifold_mixup=False,
                 ):
        'Initialization'
        
        self.x = x
        self.y = y
        self.n_samples = x.shape[0]
        self.batch_size = batch_size
        self.n_channels = n_channels

        self.dim = (x.shape[1], x.shape[1], 1)
        self.shuffle = shuffle
        self.mixup_scheme = mixup_scheme
        self.neigh = None
        self.k = k
        self.alpha = alpha
        self.local = local
        self.out_of_class = out_of_class
        self.manifold_mixup = manifold_mixup

        self.indexes = np.arange(self.n_samples)
        self.neigh = knn(x) if mixup_scheme=='knn' else None
        if self.shuffle:
          np.random.shuffle(self.indexes)
        self.on_epoch_end()

        #Parameters for images

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.n_samples / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        x, y = self.__data_generation(indexes)
        
        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_indices):
        'Generates data containing batch_size samples' 

        batch_x = self.x[batch_indices]
        batch_y = self.y[batch_indices]
        x, y = mixup(batch_x=batch_x, 
                      batch_y=batch_y, 
                      x=self.x, 
                      y=self.y, 
                      k=self.k, 
                      scheme=self.mixup_scheme, 
                      alpha=self.alpha, 
                      local=self.local, 
                      out_of_class=self.out_of_class,
                      manifold_mixup=self.manifold_mixup,
                      neigh=self.neigh
                     )

        return x, y
