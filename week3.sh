#!/bin/sh
for c1 in 'iris' 'heart disease' 'wine' 'abalone' 'arrhythmia' 'arcene' 'phishing' 'segment' 'htrue2' 'mushroom'
do
    for c3 in 'ann' 'jehmo' 'jem'
    do
        for c4 in 'none', 'random'
        do
            python main_merged.py --dataset "${c1}" --epochs 300 --train_test_ratio "${c2}" --model "${c3}" --mixup_scheme ${c4} --n_warmup 0 --n_ood 1 -ood --norm --n_warmup 50
        done
    done
done

# NOTE
# Low Stochasticity
# Experiments with ood samples