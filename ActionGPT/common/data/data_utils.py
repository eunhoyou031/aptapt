import omegaconf
import hydra
import pyrootutils
import os
import sys
import torch
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)
from transformers import AutoTokenizer
from transformers.utils import FEATURE_EXTRACTOR_NAME, get_file_from_repo
import json
from common.data.datasets import LMDBDataset_for_ActionGPT_RT1, LMDBDataset_for_ActionGPT_OXE, LMDBDataset_Mix, LMDBDataset_for_ActionGPT_CALVIN
from common.data.mix_utils import BASE_STEPSIZE, DISPLAY_KEY
from torchvision.transforms.v2 import Resize, InterpolationMode
from torch.utils.data import ConcatDataset

data_type2dataset_cls = {
    'rt1': LMDBDataset_for_ActionGPT_RT1,
    'oxe': LMDBDataset_for_ActionGPT_OXE,
    'calvin': LMDBDataset_for_ActionGPT_CALVIN,
}

def load_dataset(data_config, extra_data_config):
    if type(data_config) is str:
        data_config = omegaconf.OmegaConf.load(data_config)
        data_config = dict(data_config)
    data_config.update(extra_data_config)

    data_type = data_config.pop('data_type')
    
    if data_type == 'oxe':
        sub_data_configs = data_config.pop('sub_data_configs')
        rgb_preprocessor = Resize(
            size=data_config['rgb_shape'], 
            interpolation=InterpolationMode.BICUBIC, 
        )
        train_datasets = []
        eval_datasets = []
        train_sample_weights = []
        eval_sample_weights = []

        for sub_data_config in sub_data_configs:
            sub_data_config = dict(sub_data_config)
            data_name = sub_data_config.pop('data_name')
            weight = sub_data_config.pop('weight')
            if ('lmdb_dir' not in sub_data_config) and ('lmdb_dir' in data_config):
                sub_data_config['lmdb_dir'] = os.path.join(data_config['lmdb_dir'], data_name)
            if ('video_dir' not in sub_data_config) and ('video_dir' in data_config):
                sub_data_config['video_dir'] = os.path.join(data_config['video_dir'], data_name, DISPLAY_KEY.get(data_name, 'image'))
            step_size = max(round(BASE_STEPSIZE.get(data_name, 1) / BASE_STEPSIZE['fractal20220817_data']), 1)
            sub_data_config['skip_frame'] = data_config['skip_frame'] * step_size
            
            if 'max_skip_frame' in data_config:
                sub_data_config['max_skip_frame'] = data_config['max_skip_frame'] * step_size
                
            sub_data_config['rgb_shape'] = data_config['rgb_shape']
            sub_data_config['rgb_preprocessor'] = rgb_preprocessor

            train_dataset, eval_dataset =  load_dataset(sub_data_config, extra_data_config)
            train_datasets.append(train_dataset)
            eval_datasets.append(eval_dataset)
            train_sample_weights.append(weight)
            eval_sample_weights.append(weight)

        if data_config['weighted']:
            train_dataset = LMDBDataset_Mix(datasets=train_datasets, sample_weights=train_sample_weights)
            eval_dataset = LMDBDataset_Mix(datasets=eval_datasets, sample_weights=eval_sample_weights)
        else:
            train_dataset = ConcatDataset(train_datasets)
            eval_dataset = ConcatDataset(eval_datasets)
            
    else:
        dataset_cls = data_type2dataset_cls[data_type]
        
        if 'rgb_shape' in data_config and data_config['rgb_shape'] != [224, 224]:
            rgb_preprocessor = Resize(
                size=[224, 224],
                interpolation=InterpolationMode.BICUBIC,
            )
            data_config['rgb_preprocessor'] = rgb_preprocessor
        
        train_dataset = dataset_cls(split='train', **data_config)
        eval_dataset = dataset_cls(split='val', **data_config)
    
    return train_dataset, eval_dataset