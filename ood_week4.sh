#!/bin/sh
for c1 in 'iris' 'wine' 'heart' 'abalone' 
do
   for c3 in 'jehm' 'jehmo' 'ann' 'jem'
   do
      python main.py --dataset "${c1}" --epochs 10 --model "${c3}" --n_warmup 5 --n_ood 1
   done
done
