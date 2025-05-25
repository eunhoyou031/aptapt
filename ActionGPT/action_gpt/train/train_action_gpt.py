import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Detected kernel version")
warnings.filterwarnings("ignore", message="resume_download")

import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)
import argparse
import json
from torch.utils.data import DataLoader
import omegaconf
import hydra
from functools import partial
from transformers import AutoTokenizer
from common.models.model_utils import load_model
from common.processors.preprocessor_utils import get_rgb_preprocessor
from action_gpt.src.trainers.action_gpt_trainer import ActionGPT_Trainer
from torch.utils.data import DataLoader
from functools import partial
from common.data.data_utils import load_dataset

def main(cfg):
    action_gpt_config_path = cfg['action_gpt_config_path']
    action_gpt_config = omegaconf.OmegaConf.load(action_gpt_config_path)
    action_gpt = hydra.utils.instantiate(action_gpt_config)
    action_gpt.config = action_gpt_config

    lang_tokenizer = AutoTokenizer.from_pretrained(action_gpt_config['model_lang']['pretrained_model_name_or_path'])
    rgb_preprocessor = get_rgb_preprocessor(**cfg['rgb_preprocessor_config'])

    dataset_config_path = cfg['dataset_config_path']
    
    extra_data_config = {
        'sequence_length': action_gpt_config['sequence_length'],
        'chunk_size': action_gpt_config['action_chunk_size'],
        'prev_action_buffer_size': action_gpt_config['prev_action_buffer_size'],
        'act_dim': action_gpt_config['act_dim'],
        'do_extract_action': action_gpt_config['act_pred']
    }
    
    train_dataset, eval_dataset = load_dataset(dataset_config_path, extra_data_config)
    
    dataloader_cls = partial(
        DataLoader, 
        pin_memory=True,
        shuffle=True,
        persistent_workers=True,
        num_workers=cfg['dataloader_config']['workers_per_gpu'],
        batch_size=cfg['dataloader_config']['bs_per_gpu'],
    )
    train_dataloader = dataloader_cls(train_dataset)
    eval_dataloader = dataloader_cls(eval_dataset)
    
    trainer = ActionGPT_Trainer(
        action_gpt=action_gpt,
        action_gpt_config=action_gpt_config,
        rgb_preprocessor=rgb_preprocessor,
        lang_tokenizer=lang_tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        bs_per_gpu=cfg['dataloader_config']['bs_per_gpu'],
        **cfg['training_config']
    )

    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="/path/to/your/action_gpt_config.yaml")
    args = parser.parse_args()

    cfg = omegaconf.OmegaConf.load(args.config_path)
    main(cfg)
