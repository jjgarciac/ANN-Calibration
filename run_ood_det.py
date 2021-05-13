import os

datasets = [
            'iris',
            'heart',
            'arrhythmia',
            #'abalone',
            'wine',
            'segment',
            #'sensorless_drive',
            ]

manifolds = ['true', ]
model = ['manifold_mixup']

for i_m in model:
    for i_d in datasets:
        print('Current Method: ' + i_m + ', Current dataset: ' + i_d + '.\n')
        os.system('python main_merged.py --dataset ' + str(i_d) +
                  ' --manifold_mixup ' + '--model ' + str(i_m) +
                  ' --batch_size 64 --epochs 500 --n_ood 1'
                  ' --buffer_in False --buffer_out False')



'''
for i_m in manifolds:
    for i_d in datasets:
        print('Current Method: ' + str('manifold_mix_up') + ', Current dataset: ' + i_d + '.\n')
        os.system('python main_merged.py --dataset ' + str(i_d) + ' --manifold_mixup --epochs 300 --n_ood 1')
'''

