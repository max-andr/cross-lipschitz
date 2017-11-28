"""
This module is used when the models are trained. It restores the saved models and calculates the robustness,
measuring the minimum norm of an adversarial change that is needed to change the predicted class.

The generation of the adversarial change is done using the proposed algorithm wrt L2-norm. On top of it a
binary search is used to determine the adversarial example that lies on the decision boundary (up to desired precision).
"""
import os
import subprocess
import time
import numpy as np


async_flag = True
python_path = '/scratch/maksym/crosslipschitz/env/bin/python'
exp_name = 'docl_1hl_plain'
server = 'tenos'
if server == 'thera':  # distribute the workload between GPUs
    gpu_list = 4*[0] + 4*[1] + 4*[2] + 4*[3] + 4*[4] + 4*[5] + 4*[6] + 4*[7]
else:
    gpu_list = 4*[0] + 4*[1] + 0*[2] + 4*[3] + 4*[4] + 4*[5] + 4*[6] + 4*[7]
gpu_list = list(np.random.permutation(gpu_list))

time_start = time.time()
for dataset in ['mnist', 'cifar10']:  # ['mnist', 'cifar10', 'gtrsrb']:
    for reg_type in ['dropout']:  # ['cross_lipschitz', 'no', 'weight_decay', 'dropout']:
        reg_dir_path = 'logs/{}/{}/{}/'.format(exp_name, dataset, reg_type)
        try:
            model_names = os.listdir(reg_dir_path)
            for model_name in model_names:
                # model_path = 'models/{}/{}/{}/{}'.format(exp_name, dataset, reg_type, model_name)
                hps_list = model_name.split('test=')[1].split(' ')[1:-1]  # 1 is a test err, -1 is empty string
                hps_list_prefixed = ['--' + hp for hp in hps_list]
                hps_str_arg = ' '.join(hps_list_prefixed)

                hps_str_arg = hps_str_arg.replace('=True', '')
                hps_str_arg = hps_str_arg.replace(' --augm_flag=False', '')
                hps_str_arg = hps_str_arg.replace(' --adv_train_flag=False', '')

                gpu_number = gpu_list.pop()
                # In order to restore, we need to pass all main HPS from model_path
                command = '{} worker.py --eval_robustness --experiment_name={} --dataset={} --gpu_number={} {}'. \
                    format(python_path, exp_name, dataset, gpu_number, hps_str_arg)
                if async_flag:
                    subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
                else:
                    subprocess.call(command, shell=True, preexec_fn=os.setsid)
        except FileNotFoundError:
            print('There is no {} directory'.format(reg_dir_path))

print("Eval robustness is done in {:.2f} min".format((time.time() - time_start) / 60))


