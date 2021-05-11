import os

datasets = [
            'iris',
            'heart',
            'arrhythmia',
            'abalone',
            'wine',
            'segment',
            'sensorless_drive',
            ]

model = ['jehm', ]

for i_d in datasets:
    print('Current Method: ' + str('jehm') + ', Current dataset: ' + i_d + '.\n')
    os.system('python main_merged.py --dataset ' + str(i_d) + ' --model jehm --epochs 300 --n_warmup 0 --od_l 4.9 --od_lr 1.7 --od_n 20 --od_std .2 --n_ood 1')
