import os
'''
datasets = [
            'iris',
            'heart',
            'arrhythmia',
            #'abalone',
            'wine',
            'segment',
            #'sensorless_drive',
            ]
'''
datasets = ['phishing',
           'moon',
           'htru2',
           'mushroom',
           'arcene',
           'abalone', ]

manifolds = ['true', ]
model = ['manifold_mixup']


for i_d in datasets:
    print('Current Method: ' + 'ann' + ', Current dataset: ' + i_d + '.\n')
    os.system('python main_no_ood.py --dataset ' + str(i_d) +
              ' --model ann --mixup_scheme none'
              ' --batch_size 64 --epochs 500 --n_ood 0'
              ' --buffer_in False --buffer_out False')
