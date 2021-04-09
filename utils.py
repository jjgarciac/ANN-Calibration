import numpy as np
import data_loader
from sklearn.model_selection import train_test_split

def prepare_ood(x_train, x_val, x_test, y_train, y_val, y_test, n_ood, norm):
    n_mean = np.mean(x_train, axis=0)
    n_std = np.var(x_train, axis=0)**.5 

    idx_train_ood = np.argmax(y_train, axis=1)>n_ood
    idx_train_in = np.argmax(y_train, axis=1)<=n_ood
    idx_test_ood = np.argmax(y_test, axis=1)>n_ood
    idx_test_in = np.argmax(y_test, axis=1)<=n_ood
    idx_val_ood = np.argmax(y_val, axis=1)>n_ood
    idx_val_in = np.argmax(y_val, axis=1)<=n_ood
    x_train_ood = x_train[idx_train_ood]
    y_train_ood = y_train[idx_train_ood][:, :n_ood+1]
    x_val_ood = x_val[idx_val_ood]
    y_val_ood = y_val[idx_val_ood][:, :n_ood+1]
    x_test_ood = x_test[idx_test_ood]
    y_test_ood = y_test[idx_test_ood][:, :n_ood+1]
    
    x_ood = np.concatenate([x_train_ood, x_val_ood, x_test_ood], axis=0)
    y_ood = np.concatenate([y_train_ood, y_val_ood, y_test_ood], axis=0)

    x_train = x_train[idx_train_in]
    x_test = x_test[idx_test_in]
    x_val = x_val[idx_val_in]
    y_train = y_train[idx_train_in][:, :n_ood+1]
    y_test = y_test[idx_test_in][:, :n_ood+1]
    y_val = y_val[idx_val_in][:, :n_ood+1]

    if norm:
        print("Normalizing ood samples")
        x_odd = (x_ood - n_mean)/n_std

    return x_train, x_val, x_test, y_train, y_val, y_test, x_ood, y_ood

def update_n_ood(data, DATASET, N_OOD):
    if DATASET not in ['arcene', 'moon', 'toy_Story', 'toy_Story_ood', 'segment']:
        print(DATASET)
        y = data['labels']
        # check whether the choice of N_OOD is reasonable
        classes = np.argmax(y, axis=1)
        number_of_each_class = [(classes == ic).sum() for ic in range(int(classes.max()))]
        number_of_each_class.reverse()
        percentage_of_each_class = np.cumsum(np.array(number_of_each_class)) / np.array(number_of_each_class).sum()
        n_ood = np.where(percentage_of_each_class>=0.1)[0][0] + 1
    else:
        n_ood = int(N_OOD)

    return n_ood
