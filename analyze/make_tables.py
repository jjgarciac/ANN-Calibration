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


dict_results = {}
#####################
jem_root = '../results/results_jem.txt'
df_result = pd.read_csv(jem_root, header=0)
model = df_result.values[:, 0]
datasets = df_result.values[:, 1]
id = df_result.values[:, 2]
acc = df_result.values[:, 3]
ece = df_result.values[:, 4]
oe = df_result.values[:, 5]

for i_d in datasets:
    dict_results[i_d] = {}
    dict_results[i_d]['ACC'] = {}
    dict_results[i_d]['ECE'] = {}
    dict_results[i_d]['OE'] = {}
for i in df_result.values.shape[0]:
    dict_results[datasets[i]]['ACC'][model[i]] = acc[i]

for i_datasets in np.unique(datasets):
    dict_results[i_datasets]['ACC'] = pd.Series(dict_results[i_datasets]['ACC']).to_string()
    dict_results[i_datasets]['ECE'] = pd.Series(dict_results[i_datasets]['ECE']).to_string()
    dict_results[i_datasets]['OE'] = pd.Series(dict_results[i_datasets]['OE']).to_string()


#########################################
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
            #'toy_Story',
            #'toy_Story_ood'
            ]
mix_up_schemes = ['none', 'random', ]
models = ['none', 'random', 'manifold']
TRAIN_TEST_RATIO = 0.9
BATCH_SIZE = 16
EPOCHS = 100
LOCAL_RANDOM = True
OUT_OF_CLASS = False
MANIFOLD_MIXUP = False


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
        gdrive_rpath = '../experiments_100_epoch/'
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
            dict_results[i_d]['ACC'][i_model] = round(dict['accuracy'], 4)
            dict_results[i_d]['ECE'][i_model] = round(dict['ece_metrics'], 4)
            dict_results[i_d]['OE'][i_model] = round(dict['oe_metrics'], 4)
            del dict
        except:
            print('Lack ' + model_path)
        #dict_results[i_d]['ACC'][i_model] = round(dict['accuracy'], 4)
        #dict_results[i_d]['ECE'][i_model] = round(dict['ece_metrics'], 4)
        #dict_results[i_d]['OE'][i_model] = round(dict['oe_metrics'], 4)


    dict_results[i_d]['ACC'] = pd.Series(dict_results[i_d]['ACC']).to_string()
    dict_results[i_d]['ECE'] = pd.Series(dict_results[i_d]['ECE']).to_string()
    dict_results[i_d]['OE'] = pd.Series(dict_results[i_d]['OE']).to_string()

pd.DataFrame(dict_results).to_latex('../results/baseline', multirow=True)
pd.DataFrame.from_dict(dict_results).to_csv('../results/baseline.csv')

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
            try:
                current_improve = dict_results[i_dataset][i_metrics][i_model] - dict_results[i_dataset][i_metrics]['none']
                current_metrics[i_model][i_metrics].append(current_improve)
            except:
                print(i_model)
        MEAN = round(np.array(current_metrics[i_model][i_metrics]).mean(), 4)
        STD =  round(np.array(current_metrics[i_model][i_metrics]).std(), 4)
        current_metrics[i_model][i_metrics] = str(MEAN) + '±' + str(STD)

pd.DataFrame(current_metrics).to_latex('../results/average_improvement', multirow=True)
pd.DataFrame(current_metrics).to_csv('../results/average_improvement.csv')
