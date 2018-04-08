#!/bin/bash

python worker.py --experiment_name=advers_training --adv_train_flag --reg_type=cross_lipschitz_updated --dataset=cifar10 --nn_type=resnet --gpu_number=0 --gpu_memory=0.9 --lr=0.2 --lmbd=0.0001 --batch_size=128 --n_epochs=200
