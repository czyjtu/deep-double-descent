#!/bin/bash

for (( c=30; c<=600; c+=30 ))
do
    python3.10 train.py --steps 200000 --lr 0.1 --c $c --label_noise 0.2 
done
