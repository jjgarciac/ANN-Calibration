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
            ]
model = ['random', 'none']
for i_m in model:
    for i_d in datasets:
        print('Current Method: ' + i_m + ', Current dataset: ' + i_d + '.\n')
        os.system('python main.py --dataset ' + str(i_d) + ' --mixup_scheme ' + i_m + ' --epochs 100')

manifolds = ['true', ]
for i_m in manifolds:
    for i_d in datasets:
        print('Current Method: ' + str('manifold_mix_up') + ', Current dataset: ' + i_d + '.\n')
        os.system('python main.py --dataset ' + str(i_d) + ' --manifold_mixup' + ' --epochs 100')