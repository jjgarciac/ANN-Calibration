import os
import pandas as pd
import pickle
import ast

def open_dict_txt(dict_filename):
    file = open(dict_filename, "r")
    contents = file.read()
    dictionary = ast.literal_eval(contents)
    file.close()
    return dictionary

datasets = ['abalone',
            'arcene',
            'arrhythmia',
            'iris',
            'phishing',
            'moon',
            'sensorless_drive',
            'segment',
            'htru2',
            'heart',
            'mushroom',
            'wine',
            'toy_Story',
            'toy_Story_ood']
mix_up_schemes = [ 'none', 'random']
TRAIN_TEST_RATIO = 0.9
BATCH_SIZE = 16
EPOCHS = 10
LOCAL_RANDOM = True
OUT_OF_CLASS = False
MANIFOLD_MIXUP = False

dict_results = {}
for i_d in datasets:
    dict_results[i_d] = {}
    dict_results[i_d]['ACC'] = {}
    dict_results[i_d]['ECE'] = {}
    dict_results[i_d]['OE'] = {}
    for i_mix_up_scheme in mix_up_schemes:
        if i_mix_up_scheme == 'none':
            ALPHA = 0
        else:
            ALPHA = 0.4
        if i_mix_up_scheme == 'random':
            N_NEIGHBORS = 0
        else:
            N_NEIGHBORS = 20
        MODEL_NAME = '{}/r{}-b{}-e{}-a{}-{}-n{}-l{}-o{}{}'.format(i_d,
                                                                  TRAIN_TEST_RATIO,
                                                                  BATCH_SIZE,
                                                                  EPOCHS,
                                                                  ALPHA,
                                                                  i_mix_up_scheme,
                                                                  N_NEIGHBORS,
                                                                  1 if LOCAL_RANDOM else 0,
                                                                  1 if OUT_OF_CLASS else 0,
                                                                  "-manifold" if MANIFOLD_MIXUP else ""
                                                                  )
        gdrive_rpath = '../experiments/'
        model_path = os.path.join(gdrive_rpath, MODEL_NAME)
        try:
            list_ts = os.listdir(model_path)
            metric_file = os.path.join(gdrive_rpath, MODEL_NAME, '{}/results.txt'.format(list_ts[0])) # use the 1st time point
            dict = open_dict_txt(metric_file)
        except:
            print('Lack ' + model_path)
        dict_results[i_d]['ACC'][i_mix_up_scheme] = dict['accuracy']
        dict_results[i_d]['ECE'][i_mix_up_scheme] = dict['ece_metrics']
        dict_results[i_d]['OE'][i_mix_up_scheme] = dict['oe_metrics']

pd.DataFrame(dict_results).to_csv('../results/baseline.csv')


