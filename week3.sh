#!/bin/sh
for c1 in 'iris' 'segment' 'arrythmia' 'arcene' 'phishing' 'sensorless_drive' 'abalone' 'htrue2' 'heart disease' 'mushroom' 'wine' 
do
    for c2 in .1 .01 .001
    do
        for c3 in 'jem' 'jemo'
	do
            python main.py --dataset "${c1}" --epochs 100 --od_l "${c2}" --model "${c3}" --n_warmup 50
        done
    done
done
