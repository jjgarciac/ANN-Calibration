import os

datasets = ['sensorless_drive',
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

manifolds = ['true', ]
model = ['none', ]

for i_m in model:
    for i_d in datasets:
        print('Current Method: ' + i_m + ', Current dataset: ' + i_d + '.\n')
        os.system('python main_ood.py --dataset ' + str(i_d) + ' --mixup_scheme ' + i_m + ' --epochs 100 --n_ood 1')
