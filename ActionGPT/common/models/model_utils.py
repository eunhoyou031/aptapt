import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)
import omegaconf
import hydra
import os
import sys
import torch
from action_gpt.src.models.action_gpt_policy_wrapper import ActionGPT_PolicyWrapper
from transformers import AutoTokenizer
from transformers.utils import FEATURE_EXTRACTOR_NAME, get_file_from_repo
from common.processors.preprocessor_utils import get_model_vision_basic_config
import json

def load_model(pretrained_path):
    config_path = os.path.join(pretrained_path, "config.yaml")
    checkpoint_path = os.path.join(pretrained_path, "pytorch_model.bin")

    config = omegaconf.OmegaConf.load(config_path)
    model = hydra.utils.instantiate(config)
    model.config = config

    missing_keys, unexpected_keys = model.load_state_dict(torch.load(checkpoint_path), strict=False)
    missing_root_keys = set([k.split(".")[0] for k in missing_keys])
    print('load ', checkpoint_path, '\nmissing ', missing_root_keys, '\nunexpected ', unexpected_keys)

    return model

def load_action_gpt_policy(args):
    print(f"loading Action-GPT from {args.action_gpt_path} ...")
    action_gpt = load_model(args.action_gpt_path)
    action_gpt_config = action_gpt.config

    lang_tokenizer = AutoTokenizer.from_pretrained(action_gpt_config['model_lang']['pretrained_model_name_or_path'])
    model_vision_basic_config = get_model_vision_basic_config(action_gpt_config['model_vision']['pretrained_model_name_or_path'])

    variant = {
        'test_chunk_size': args.test_chunk_size,
        'is_gripper_binary': args.is_gripper_binary,
        'use_temporal_ensemble': args.use_temporal_ensemble,
        'act_dim': action_gpt_config['act_dim'],
        'seq_len': action_gpt_config['sequence_length'],
        'chunk_size': action_gpt_config['chunk_size'],
        'prev_action_buffer_size': action_gpt_config.get('prev_action_buffer_size', 10),
        'pred_discrete_arm_action': action_gpt_config.get('pred_discrete_arm_action', False)
    }
    variant.update(model_vision_basic_config)

    policy_wrapper = ActionGPT_PolicyWrapper(
        policy=action_gpt,
        variant=variant,
        lang_tokenizer=lang_tokenizer
    )

    return policy_wrapper