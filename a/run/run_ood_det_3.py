import os

datasets = [
            'segment',
            #'htru2',
            'heart',
            'mushroom',
            'wine',
            'arrhythmia',
            'iris',
            'sensorless_drive',
            #'phishing',
            #'moon',
            # 'abalone'
            ]

#manifolds = ['true', ]
model = ['random', ]

for i_m in model:
    for i_d in datasets:
        print('Current Method: ' + i_m + ', Current dataset: ' + i_d + '.\n')
        os.system('python main_ood.py --dataset ' + str(i_d) + ' --mixup_scheme ' + i_m + ' --epochs 100 --n_ood 1')
