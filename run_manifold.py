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
            'wine',]

datasets = ['toy_Story', 'toy_Story_ood']
manifolds = ['true', ]

for i_m in manifolds:
    for i_d in datasets:
        print('Current Method: ' + str('manifold_mix_up') + ', Current dataset: ' + i_d + '.\n')
        os.system('python main.py --dataset ' + str(i_d) + ' --manifold_mixup')
