import os
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
model = ['random', 'none']
for i_m in model:
    for i_d in datasets:
        print('Current Method: ' + i_m + ', Current dataset: ' + i_d + '.\n')
        os.system('python main.py --dataset ' + str(i_d) + ' --mixup_scheme ' + i_m)