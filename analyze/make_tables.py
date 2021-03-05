import os
import pandas as pd
import pickle
import ast
import numpy as np

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
mix_up_schemes = ['none', 'random', ]
models = ['none', 'random', 'manifold']
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

    for i_model in models:
        if i_model == 'none':
            ALPHA = 0
            N_NEIGHBORS = 20
            i_mix_up_scheme = i_model
            MANIFOLD_MIXUP = False
        elif i_model == 'random':
            ALPHA = 0.4
            N_NEIGHBORS = 0
            i_mix_up_scheme = i_model
            MANIFOLD_MIXUP = False
        else:
            ALPHA = 0.4
            N_NEIGHBORS = 0
            i_mix_up_scheme = 'random'
            MANIFOLD_MIXUP = True


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
            for i in range(len(list_ts)):
               metric_file = os.path.join(gdrive_rpath, MODEL_NAME, '{}/results.txt'.format(list_ts[i])) # use the 1st time point
               if os.path.exists(metric_file):
                   dict = open_dict_txt(metric_file)
                   break
               else:
                   continue
        except:
            print('Lack ' + model_path)
        dict_results[i_d]['ACC'][i_model] = round(dict['accuracy'], 4)
        dict_results[i_d]['ECE'][i_model] = round(dict['ece_metrics'], 4)
        dict_results[i_d]['OE'][i_model] = round(dict['oe_metrics'], 4)

    dict_results[i_d]['ACC'] = pd.Series(dict_results[i_d]['ACC'])#.to_string()
    dict_results[i_d]['ECE'] = pd.Series(dict_results[i_d]['ECE'])#.to_string()
    dict_results[i_d]['OE'] = pd.Series(dict_results[i_d]['OE'])#.to_string()

#pd.DataFrame(dict_results).to_latex('../results/baseline', multirow=True)
#pd.DataFrame.from_dict(dict_results).to_csv('../results/baseline.csv')

datasets_ave = datasets = ['abalone',
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
            'wine',]

current_metrics = {}
for i_model in ['random', 'manifold']:
    current_metrics[i_model] = {}
    for i_metrics in ['ACC', 'ECE', 'OE']:
        current_metrics[i_model][i_metrics] = []
        for i_dataset in datasets:
            current_improve = dict_results[i_dataset][i_metrics][i_model] - dict_results[i_dataset][i_metrics]['none']
            current_metrics[i_model][i_metrics].append(current_improve)
        MEAN = round(np.array(current_metrics[i_model][i_metrics]).mean(), 4)
        STD =  round(np.array(current_metrics[i_model][i_metrics]).std(), 4)
        current_metrics[i_model][i_metrics] = str(MEAN) + '±' + str(STD)

pd.DataFrame(current_metrics).to_latex('../results/average_improvement', multirow=True)
pd.DataFrame(current_metrics).to_csv('../results/average_improvement.csv')
