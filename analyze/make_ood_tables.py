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


DATASETS = ['sensorless_drive',
            'segment',
            #'htru2',
            'heart',
            'mushroom',
            'wine',
            'arrhythmia',
            'iris',
            #'phishing',
            #'moon',
'abalone',
]

mix_up_schemes = ['none', 'random', ]
models = ['none', 'random', 'manifold']
TRAIN_TEST_RATIO = 0.9
BATCH_SIZE = 16
EPOCHS = 100
LOCAL_RANDOM = True
OUT_OF_CLASS = False
MANIFOLD_MIXUP = False
N_OOD=1

dict_results = {}
dict_results['ACC'] = {}
dict_results['ECE'] = {}
dict_results['OE'] = {}
dict_results['AUC'] = {}
for i_d in DATASETS:
    dict_results['ACC'][i_d] = {}
    dict_results['ECE'][i_d] = {}
    dict_results['OE'][i_d] = {}
    dict_results['AUC'][i_d] = {}
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


        MODEL_NAME = '{}/r{}-b{}-e{}-a{}-{}-n{}-l{}-o{}{}-no{}'.format(i_d,
                                                                  TRAIN_TEST_RATIO,
                                                                  BATCH_SIZE,
                                                                  EPOCHS,
                                                                  ALPHA,
                                                                  i_mix_up_scheme,
                                                                  N_NEIGHBORS,
                                                                  1 if LOCAL_RANDOM else 0,
                                                                  1 if OUT_OF_CLASS else 0,
                                                                  "-manifold" if MANIFOLD_MIXUP else "",
                                                                  N_OOD,)
        gdrive_rpath = '../experiments_ood/'
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
            dict_results['ACC'][i_d][i_model] = round(dict['accuracy'], 4)
            dict_results['ECE'][i_d][i_model] = round(dict['ece_metrics'], 4)
            dict_results['OE'][i_d][i_model] = round(dict['oe_metrics'], 4)
            dict_results['AUC'][i_d][i_model] = round(dict['auc_of_ood'], 4)
            del dict
        except:
            dict_results['ACC'][i_d][i_model] = 'XXX' #round(dict['accuracy'], 4)
            dict_results['ECE'][i_d][i_model] = 'XXX' #round(dict['ece_metrics'], 4)
            dict_results['OE'][i_d][i_model] = 'XXX' #round(dict['oe_metrics'], 4)
            dict_results['AUC'][i_d][i_model] = 'XXX'
            print('Lack ' + model_path)
        #dict_results[i_d]['ACC'][i_model] = round(dict['accuracy'], 4)
        #dict_results[i_d]['ECE'][i_model] = round(dict['ece_metrics'], 4)
        #dict_results[i_d]['OE'][i_model] = round(dict['oe_metrics'], 4)


    #dict_results[i_d]['ACC'] = pd.Series(dict_results[i_d]['ACC']).to_string()
    #dict_results[i_d]['ECE'] = pd.Series(dict_results[i_d]['ECE']).to_string()
    #dict_results[i_d]['OE'] = pd.Series(dict_results[i_d]['OE']).to_string()

'''
#####################
jem_root = '../results/results_jem.txt'
df_result = pd.read_csv(jem_root, header=0)
model = df_result.values[:, 0]
datasets = df_result.values[:, 1]
id = df_result.values[:, 2]
acc = df_result.values[:, 3]
ece = df_result.values[:, 4]
oe = df_result.values[:, 5]

###########################################
for i in range(df_result.values.shape[0]):
    current_dataset = datasets[i].replace(' ', '')
    dict_results['ACC'][current_dataset][model[i]] = acc[i]
    dict_results['ECE'][current_dataset][model[i]] = ece[i]
    dict_results['OE'][current_dataset][model[i]] = oe[i]
#for i_D in DATASETS:
#    dict_results[i_D]['ACC'] = pd.Series(dict_results[i_D]['ACC']).to_string()
#    dict_results[i_D]['ECE'] = pd.Series(dict_results[i_D]['ECE']).to_string()
#    dict_results[i_D]['OE'] = pd.Series(dict_results[i_D]['OE']).to_string()
##############################################
'''
for i_metrics in ['ACC', 'ECE', 'OE', 'AUC']:
    pd.DataFrame(dict_results[i_metrics]).T.to_latex('../results/ood_baseline_' + i_metrics, multirow=True)
    pd.DataFrame.from_dict(dict_results[i_metrics]).T.to_csv('../results/ood_baseline_' + i_metrics + '.csv')


# 2. calculating average
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
for i_metrics in ['ACC', 'ECE', 'OE']:
    current_metrics[i_metrics] = {}
    for i_model in ['random', 'manifold', 'jem', 'jemo']:
        current_metrics[i_metrics][i_model] = []
        for i_dataset in datasets:
            try:
                current_improve = dict_results[i_metrics][i_dataset][i_model] - dict_results[i_metrics][i_dataset]['none']
                current_metrics[i_metrics][i_model].append(current_improve)
            except:
                print(i_model)
        MEAN = round(np.array(current_metrics[i_metrics][i_model]).mean(), 4)
        STD =  round(np.array(current_metrics[i_metrics][i_model]).std(), 4)
        current_metrics[i_metrics][i_model] = str(MEAN) + '±' + str(STD)

pd.DataFrame(current_metrics).to_latex('../results/average_improvement', multirow=True)
pd.DataFrame(current_metrics).to_csv('../results/average_improvement.csv')
