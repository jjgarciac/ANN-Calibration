import numpy as np

def prepare_ood(x_train, x_val, x_test, y_train, y_val, y_test, n_ood):
    idx_train_ood = np.argmax(y_train, axis=1)>n_ood
    idx_train_in = np.argmax(y_train, axis=1)<=n_ood
    idx_test_ood = np.argmax(y_test, axis=1)>n_ood
    idx_test_in = np.argmax(y_test, axis=1)<=n_ood
    idx_val_ood = np.argmax(y_val, axis=1)>n_ood
    idx_val_in = np.argmax(y_val, axis=1)<=n_ood
    x_train_ood = x_train[idx_train_ood]
    y_train_ood = y_train[idx_train_ood][:, n_ood+1:]
    x_val_ood = x_val[idx_val_ood]
    y_val_ood = y_val[idx_val_ood][:, n_ood+1:]
    x_test_ood = x_test[idx_test_ood]
    y_test_ood = y_test[idx_test_ood][:, n_ood+1:]
    
    x_ood = np.concatenate([x_train_ood, x_val_ood, x_test_ood], axis=0)
    y_ood = np.concatenate([y_train_ood, y_val_ood, y_test_ood], axis=0)

    x_train = x_train[idx_train_in]
    x_test = x_test[idx_test_in]
    x_val = x_val[idx_val_in]
    y_train = y_train[idx_train_in][:, :n_ood+1]
    y_test = y_test[idx_test_in][:, :n_ood+1]
    y_val = y_val[idx_val_in][:, :n_ood+1]

    return x_train, x_val, x_test, y_train, y_val, y_test, x_ood, y_ood
