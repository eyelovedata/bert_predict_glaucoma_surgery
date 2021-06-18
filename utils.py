import os
import random
import numpy as np

import torch
import json

def set_seed(seed):
    """Set all seeds to make results reproducible (deterministic mode).
       When seed is None, disables deterministic mode.
    :param seed: an integer to your choosing
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

def mkdir_for_saving_model(model_saving_dir):
    if not os.path.exists(model_saving_dir):
        os.makedirs(model_saving_dir)
        print(f'create new folder {model_saving_dir} to save the finetuned model and results')
    else:
        print(f'{model_saving_dir} folder exists')

def convert_config_log_to_json(model_saving_dir):
    """
    This function converted the config_py_setting.log into config_py_setting.json
    :param model_saving_dir: dir for log and saving the converted json
    """
    config_py_setting_mapping = {}
    location = model_saving_dir
    with open(os.path.join(location, 'config_py_setting.log')) as fp:
        line = fp.readline()
        while line:
            param_name = line[12:].strip().split(' = ')[0]
            param_val = line[12:].strip().split(' = ')[1]
            config_py_setting_mapping[param_name] = param_val
            line = fp.readline()

    json_file = json.dumps(config_py_setting_mapping)
    file_path = location + '/config_py_setting.json'
    f = open(file_path, 'w')
    f.write(json_file)
    f.close()
    print(f'config_py_setting.log was converted into json format and saved in {file_path}')