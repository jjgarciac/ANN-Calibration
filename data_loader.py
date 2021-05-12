import pandas as pd
import urllib
import numpy as np
from scipy.io import arff

from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

from sklearn.datasets import make_moons
from sklearn.datasets import make_s_curve
from sklearn.preprocessing import OneHotEncoder

def load_phising_website_data():
  url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff'
  train_data = urllib.request.urlopen(url)
  columns = []
  rows = []
  len_attr = len('@attribute')
  data = False
  for line in train_data:
      decoded_line = line.decode("utf-8")
      if not data:
          if decoded_line.startswith('@attribute '):
              col_name = decoded_line[len_attr:].split()[0]
              columns.append(col_name)
          elif decoded_line.startswith('@data'):
              data = True
      else:
          rows.append([int(i) for i in decoded_line.strip().split(',')])

  data = pd.DataFrame(data=rows, columns=columns)  

  return {
          'features':  data[[c for c in data.columns if c != 'Result']],
          'labels': OneHotEncoder().fit_transform(X=data[['Result']]).toarray()
      }


def load_htru2_data():
  resp = urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/00372/HTRU2.zip')
  zipfile = ZipFile(BytesIO(resp.read()))
  rows = []
  for row in zipfile.read('HTRU_2.csv').decode().split('\r'):
      rows.append([float(i) for i in row.split(',')])

  data = pd.DataFrame(rows, 
              columns=['Mean_integrated_profile',
                      'Std_integrated_profile',
                      'kurtosis_integrated_profile',
                        'Skewness_integrated_profile',
                        'Mean_DM-SNR_curve',
                        'Std_DM-SNR_curve',
                        'Kurtosis_DM-SNR_curve',
                        'Skewness_DM-SNR_curve',
                        'Class'
                      ])
  return {
      'features': data[[c for c in data.columns if c != 'Class']],
      'labels': OneHotEncoder().fit_transform(X=data[['Class']]).toarray()
  }

def make_toy_Story(n_samples_per_class = 100, ood=False):
  mu1 = [.0, 1]
  mu2 = [.0, -1]
  cov = [[.05, 0], [0, .05]]
  X, color = make_s_curve(n_samples_per_class*3, random_state=0)
  s_labels = 0*(X[:,2]<-.5) + 1*(-.5<=X[:,2])*(X[:,2]<1) + 2*(X[:,2]>=1)
  n1 = np.random.multivariate_normal(mu1, cov, n_samples_per_class)
  n2 = np.random.multivariate_normal(mu2, cov, n_samples_per_class)
  X = np.concatenate([X[:,[0,2]], n1, n2])
  Y = np.concatenate([s_labels, [3]*n_samples_per_class, [4]*n_samples_per_class])
  p = np.random.permutation(len(X))
  return X[p], Y[p]

def make_toy_Story_with_ood_class(n_samples_per_class = 100):
  mu1 = [.0, 1]
  mu2 = [.0, -1]
  cov = [[.05, 0], [0, .05]]
  X, color = make_s_curve(n_samples_per_class*3, random_state=0)
  s_labels = 0*(X[:,2]<-.5) + 1*(-.5<=X[:,2])*(X[:,2]<1) + 2*(X[:,2]>=1)
  n1 = np.random.multivariate_normal(mu1, cov, n_samples_per_class)
  n2 = np.random.multivariate_normal(mu2, cov, n_samples_per_class)
  u1 = np.random.uniform(0, 2*np.pi, [int(n_samples_per_class*5), 1])
  ood_s = np.concatenate([2.2*np.cos(u1), 2.2*np.sin(u1)], axis=1)
  X = np.concatenate([X[:,[0,2]], n1, n2, ood_s])
  Y = np.concatenate([s_labels, [3]*n_samples_per_class, [4]*n_samples_per_class, [5]*int(n_samples_per_class*5)])
  p = np.random.permutation(len(X))
  return X[p], Y[p]

def load(dname='abalone', 
         url='https://archive.ics.uci.edu/ml/machine-learning-databases',
         n_train=10000,
         n_test=1000,
         train_noise=0.1,
         test_noise=0.1,
         ood=False):
  """
  Load dataset.
  dname, str: heart, abalone, arcene, arrhythmia, iris, phishing, htru2
  """
  if dname == 'segment':
      train_val_data = pd.read_csv('{}/image/segmentation.data'.format(url),
                         skiprows=[0, 1, 2],
                         header=0, )
      test_data = pd.read_csv('{}/image/segmentation.test'.format(url),
                         skiprows=[0, 1, 2],
                         header=0)
      ec = OneHotEncoder()
      x_train = train_val_data
      y_train = ec.fit_transform(np.array(train_val_data.index).reshape(-1, 1)).toarray()
      x_test = test_data
      y_test = ec.fit_transform(np.array(test_data.index).reshape(-1, 1)).toarray()

      return {
              'x_train': x_train,
              'y_train': y_train,
              'x_val': x_test,
              'y_val': y_test,
              }

  if dname == 'sensorless_drive':
      data = pd.read_csv('{}/00325/Sensorless_drive_diagnosis.txt'.format(url),
                         delim_whitespace=True,
                         header=None)
      return {
              'features': data[range(48)],
              'labels': OneHotEncoder().fit_transform(data[[48]]).toarray()
              }

  if dname == 'heart':
      data = pd.read_csv('{}/heart-disease/processed.cleveland.data'.format(url),
                          sep=',',
                          header=None,
                        names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
                        )
    
      return {
              'features': data[[c for c in data.columns if c != 'num']],
              'labels': OneHotEncoder().fit_transform(data[['num']]).toarray()
              }
    
  if dname == 'mushroom':
      data = pd.read_csv('{}/mushroom/agaricus-lepiota.data'.format(url),
                          sep=',', 
                          header=None,
                        )
      return {
          'features': data[[c for c in data.columns if c != 0]],
          'labels': OneHotEncoder().fit_transform(data[[0]]).toarray()
      }

  if dname == 'wine':
      red_wine = pd.read_csv('{}/wine-quality/winequality-red.csv'.format(url), 
                             sep=';', 
                            )
      white_wine = pd.read_csv('{}/wine-quality/winequality-white.csv'.format(url), 
                             sep=';', 
                            )
  
      data = pd.concat([red_wine, white_wine], axis=0)
      return {
          'features': data[[c for c in data.columns if c != 'quality']],
          'labels': OneHotEncoder().fit_transform(data[['quality']]).toarray()
      }

  if dname == 'abalone':
      url = "{}/{}/{}.data".format(url, dname, dname)
      data = pd.read_csv(url, 
                          header=None, 
                          names=['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight','Rings'])
  
      return {
          'features': data[[c for c in data.columns if c != 'Rings']],
          'labels': OneHotEncoder().fit_transform(data[['Rings']]).toarray()
      }
  
  if dname == 'arcene':
      x_train = pd.read_csv('{}/{}/{}/arcene_train.data'.format(url, dname, dname.upper()),
                                  sep=' ', 
                                  header=None
                                )
      
      x_test = pd.read_csv('{}/{}/{}/arcene_test.data'.format(url, dname, dname.upper()),
                                  sep=' ', 
                                  header=None
                                )
      
      y_train = pd.read_csv('{}/{}/{}/arcene_train.labels'.format(url, dname, dname.upper()),
                      sep=' ', 
                      header=None
                    )
      
      x_val = pd.read_csv('{}/{}/{}/arcene_valid.data'.format(url, dname, dname.upper()),
                      sep=' ', 
                      header=None
                    )
      
      y_val = pd.read_csv('{}/{}/arcene_valid.labels'.format(url, dname),
                      sep=' ', 
                      header=None
                    )
      

      ec = OneHotEncoder()
      y_train_enc = ec.fit_transform(X=y_train).toarray()
      y_val_enc = ec.transform(X=y_val).toarray()

      return {
              'x_train': x_train.drop(columns=[10000]),
              'y_train': y_train_enc,
              'x_val': x_val.drop(columns=[10000]),
              'y_val': y_val_enc,
              'x_test': x_test.drop(columns=[10000])
              }

  if dname == 'arrhythmia':
      data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data", 
                          header=None
                        )
      return {
          'features': data[[i for i in range(279)]],
          'labels': OneHotEncoder().fit_transform(data[[279]]).toarray()
      }
      
  if dname == 'iris':
      data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                          header=None,
                          names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])
      
      ys = data['class'].unique()
      idx = {k:i for i, k in enumerate(ys)}
      data['class'] = data['class'].apply(lambda x: idx[x]).values
      
      return {
          'features': data[[c for c in data.columns if c != 'class']],
          'labels': OneHotEncoder().fit_transform(data[['class']]).toarray(),
      }
      
      
  if dname == 'phishing':
      return load_phising_website_data()
      
  
  if dname == 'htru2':
      return load_htru2_data()

  if dname == 'moon':
    x_train, y_train =  make_moons(n_train, noise=train_noise)
    y_train = y_train.reshape(-1,1)
    enc = OneHotEncoder(sparse=False).fit(y_train)
    y_train = enc.transform(y_train)

    x_test, y_test = make_moons(n_test, noise=test_noise)
    y_test = y_test.reshape(-1,1)
    y_test = enc.transform(y_test)

    return { 
            'x_train': x_train,
            'y_train': y_train,
            'x_val': x_test,
            'y_val':y_test,
           }

  if dname == 'toy_story':
    x_train, y_train = make_toy_Story(int(n_train/5)+1)
    y_train = y_train.reshape(-1,1)
    enc = OneHotEncoder(sparse=False).fit(y_train)
    y_train = enc.transform(y_train)

    x_test, y_test = make_toy_Story(int(n_test/5)+1)
    y_test = y_test.reshape(-1,1)
    enc = OneHotEncoder(sparse=False).fit(y_test)
    y_test = enc.transform(y_test)
    
    if ood:
        print("Adding OOD samples to data")
        u1 = np.random.uniform(0, 2*np.pi, [int(n_train*2)+1, 1])
        ood_samples = np.concatenate([2.2*np.cos(u1), 2.2*np.sin(u1)], axis=1)
        ood_samples = ood_samples + np.random.normal(0, .125, size=ood_samples.shape)
        x_train = np.concatenate([x_train, ood_samples], axis=0)
        y_train = np.concatenate([y_train, [[0.2]*5]*(int(n_train*2)+1)], axis=0)
        p = np.random.permutation(len(x_train))
        x_train = x_train[p]
        y_train = y_train[p]
    return {
           'x_train': x_train,
           'y_train': y_train,
           'x_val': x_test,
           'y_val':y_test,
      }

  if dname == 'mnist':
    df = pd.read_csv('./img_datasets/MNIST.csv')
    
    n_data = int(df.iloc[df.shape[0] - 3][1])
    input_dims = [28, 28, 1]
    input_dim = int(np.prod(input_dims))
   
    # if len(input_dims) == 3:
    #     if input_dims[2] == 3:
    #         transform = transforms.Compose([transforms.ToPILImage(),
    #                                             transforms.RandomHorizontalFlip(),
    #                                             transforms.RandomCrop(32, 4),
    #                                             transforms.RandomRotation(degrees=15)])
    
    X, Y = df.values[:n_data, 1:input_dim + 1], df.values[:n_data, input_dim + 1: input_dim + 2]
    X = X/255.
    X = np.reshape(X, [-1, input_dims[0], input_dims[1], input_dims[2]])
    enc = OneHotEncoder(sparse=False).fit(Y)
    Y = enc.transform(Y)
    p = np.random.permutation(X.shape[0])[:n_train] 
    return {
           'features': X[p],
           'labels': Y[p]}

  if dname == 'kmnist':
    df = pd.read_csv('./img_datasets/MNIST.csv')
    
    n_data = int(df.iloc[df.shape[0] - 3][1])
    input_dims = [28, 28, 1]
    input_dim = int(np.prod(input_dims))
   
    #if len(input_dims) == 3:
    #    if input_dims[2] == 3:
    #        transform = transforms.Compose([transforms.ToPILImage(),
    #                                            transforms.RandomHorizontalFlip(),
    #                                            transforms.RandomCrop(32, 4),
    #                                            transforms.RandomRotation(degrees=15)])
    
    X, Y = df.values[:n_data, 1:input_dim + 1], df.values[:n_data, input_dim + 1: input_dim + 2]
    X = X/255.
    X = np.reshape(X, [-1, input_dims[0], input_dims[1], input_dims[2]])
    enc = OneHotEncoder(sparse=False).fit(Y)
    Y = enc.transform(Y)
    
    p = np.random.permutation(X.shape[0])[:n_train] 
    return {
           'features': X[p],
           'labels': Y[p]}

  if dname == 'f-mnist':
    df = pd.read_csv('./img_datasets/MNIST.csv')
    
    n_data = int(df.iloc[df.shape[0] - 3][1])
    input_dims = [28, 28, 1]
    input_dim = int(np.prod(input_dims))
   
    # if len(input_dims) == 3:
    #     if input_dims[2] == 3:
    #         transform = transforms.Compose([transforms.ToPILImage(),
    #                                             transforms.RandomHorizontalFlip(),
    #                                             transforms.RandomCrop(32, 4),
    #                                             transforms.RandomRotation(degrees=15)])
    
    X, Y = df.values[:n_data, 1:input_dim + 1], df.values[:n_data, input_dim + 1: input_dim + 2]
    X = X/255.
    X = np.reshape(X, [-1, input_dims[0], input_dims[1], input_dims[2]])
    enc = OneHotEncoder(sparse=False).fit(Y)
    Y = enc.transform(Y)
    
    p = np.random.permutation(X.shape[0])[:n_train] 
    return {
           'features': X[p],
           'labels': Y[p]}

  if dname == 'cifar10':
    df = pd.read_csv('./img_datasets/CIFAR10.csv.')
    y_train = y_train.reshape(-1,1)
    enc = OneHotEncoder(sparse=False).fit(y_train)
    y_train = enc.transform(y_train)

    x_test, y_test = make_toy_Story_with_ood_class(int(n_test/5)+1)
    y_test = y_test.reshape(-1,1)
    enc = OneHotEncoder(sparse=False).fit(y_test)
    y_test = enc.transform(y_test)
    
    return {
           'x_train': x_train,
           'y_train': y_train,
           'x_val': x_test,
           'y_val':y_test,
      }
 
  if dname == 'svhn':
    df = pd.read_csv('./img_datasets/')
    y_train = y_train.reshape(-1,1)
    enc = OneHotEncoder(sparse=False).fit(y_train)
    y_train = enc.transform(y_train)

    x_test, y_test = make_toy_Story_with_ood_class(int(n_test/5)+1)
    y_test = y_test.reshape(-1,1)
    enc = OneHotEncoder(sparse=False).fit(y_test)
    y_test = enc.transform(y_test)
    
    return {
           'x_train': x_train,
           'y_train': y_train,
           'x_val': x_test,
           'y_val':y_test,
      }

  if dname == 'toy_Story_ood':
    x_train, y_train = make_toy_Story_with_ood_class(int(n_train/5)+1)
    y_train = y_train.reshape(-1,1)
    enc = OneHotEncoder(sparse=False).fit(y_train)
    y_train = enc.transform(y_train)

    x_test, y_test = make_toy_Story_with_ood_class(int(n_test/5)+1)
    y_test = y_test.reshape(-1,1)
    enc = OneHotEncoder(sparse=False).fit(y_test)
    y_test = enc.transform(y_test)
    
    return {
           'x_train': x_train,
           'y_train': y_train,
           'x_val': x_test,
           'y_val':y_test,
      }

  if dname == 'toy_Story_ood':
    x_train, y_train = make_toy_Story_with_ood_class(int(n_train/5)+1)
    y_train = y_train.reshape(-1,1)
    enc = OneHotEncoder(sparse=False).fit(y_train)
    y_train = enc.transform(y_train)

    x_test, y_test = make_toy_Story_with_ood_class(int(n_test/5)+1)
    y_test = y_test.reshape(-1,1)
    enc = OneHotEncoder(sparse=False).fit(y_test)
    y_test = enc.transform(y_test)
    
    return {
           'x_train': x_train,
           'y_train': y_train,
           'x_val': x_test,
           'y_val':y_test,
      }



def prepare_inputs(x_train, x_test=None):
  numeric_cols = list(x_train._get_numeric_data().columns)
  categorical_cols = set(x_train.columns) - set(numeric_cols)

  train_features = [x_train[numeric_cols]]
  if x_test is not None:
    test_features = [x_test[numeric_cols]]

  for cc in categorical_cols:
    cats = x_train[cc].unique()
    oe = OneHotEncoder()
    train_enc = oe.fit_transform(x_train[[cc]]).toarray()
    n = train_enc.shape[-1]
    columns=['{}_{}'.format(cc, i) for i in range(n)]

    train_features.append(pd.DataFrame(train_enc, 
                                         columns=columns))

    if x_test is not None:
      test_enc = oe.transform(x_test[[cc]]).toarray()
      test_features.append(pd.DataFrame(test_enc, 
                                          columns=columns))
    
    
  if x_test is not None:
    return pd.concat(train_features, axis=1).values, pd.concat(test_features, axis=1).values
  return pd.concat(train_features, axis=1).values
