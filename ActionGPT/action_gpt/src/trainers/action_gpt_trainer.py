import os
from time import time
import torch
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
from common.data.datasets import DataPrefetcher
from action_gpt.src.trainers.trainer_utils import cross_entropy, masked_loss
import omegaconf
from glob import glob
import shutil
import wandb

class ActionGPT_Trainer:
    def __init__(
        self,
        action_gpt,
        action_gpt_config,
        rgb_preprocessor,
        lang_tokenizer,
        train_dataloader,
        eval_dataloader,
        save_path,
        save_epochs=1,
        save_steps=10000,
        num_epochs=20,
        print_steps=100,
        lr_max=0.0001,
        weight_decay=0.0001,
        num_warmup_epochs=1,
        gradient_accumulation_steps=4,
        resume_ckpt_path=None,
        bs_per_gpu=32,
        max_epoch=None,
    ):
        try:
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
            accelerator = Accelerator(
                gradient_accumulation_steps=gradient_accumulation_steps,
                kwargs_handlers=[ddp_kwargs]
            )
            self.accelerator = accelerator
            
        except Exception as e:
            print(f"Error during trainer initialization: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        if resume_ckpt_path is not None:
            self.print(f"resuming ActionGPT from {resume_ckpt_path} ...")

            current_model_dict = action_gpt.state_dict()
            resume_model_dict = torch.load(os.path.join(resume_ckpt_path, 'pytorch_model.bin'), map_location='cpu')

            mismatched_param_names = []
            filtered_state_dict = {}

            for name, param in resume_model_dict.items():
                if name in current_model_dict and current_model_dict[name].shape != param.shape:
                    mismatched_param_names.append(name)
                else:
                    filtered_state_dict[name] = param

            missing_keys, unexpected_keys = action_gpt.load_state_dict(filtered_state_dict, strict=False)
            missing_root_keys = set([k.split(".")[0] for k in missing_keys])
            self.print('load ', resume_ckpt_path, '\nmissing ', missing_root_keys, '\nunexpected ', unexpected_keys, '\nmismatched ', mismatched_param_names)
        
        optimizer = torch.optim.AdamW(action_gpt.parameters(), lr=lr_max, weight_decay=weight_decay, fused=True)
        total_prints_per_epoch = len(train_dataloader.dataset) // (print_steps * bs_per_gpu * accelerator.num_processes)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=min(num_warmup_epochs*total_prints_per_epoch, 5000000 // (print_steps * bs_per_gpu * accelerator.num_processes)),
            num_training_steps=num_epochs*total_prints_per_epoch,
        )
        
        action_gpt, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            action_gpt, optimizer, train_dataloader, eval_dataloader, 
            device_placement=[True, True, False, False]
        )
        
        self.writer = SummaryWriter(os.path.join(save_path, 'logs'))
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.total_prints_per_epoch = total_prints_per_epoch
        self.action_gpt = action_gpt
        self.action_gpt_config = action_gpt_config
        self.optimizer = optimizer
        
        self.train_prefetcher = DataPrefetcher(train_dataloader, self.device, lang_tokenizer=lang_tokenizer)
        self.eval_prefetcher = DataPrefetcher(eval_dataloader, self.device, lang_tokenizer=lang_tokenizer)
        
        self.act_dim = action_gpt_config.get('act_dim', 7)
        self.sequence_length = action_gpt_config.get('sequence_length', 1)
        self.action_chunk_size = action_gpt_config.get('chunk_size', 5)
        self.prev_action_buffer_size = action_gpt_config.get('prev_action_buffer_size', 10)
        self.pred_binary_gripper_action = action_gpt_config.get('is_gripper_binary', True)
        self.pred_discrete_arm_action = action_gpt_config.get('is_gripper_binary', False)
        
        self.rgb_preprocessor = rgb_preprocessor.to(self.device)
        self.lang_tokenizer = lang_tokenizer
        self.save_path = save_path
        self.save_epochs = save_epochs
        self.save_steps = save_steps
        self.max_epoch = max_epoch
        self.num_epochs = num_epochs
        self.print_steps = print_steps
        self.bs_per_gpu = bs_per_gpu
        
        
        if self.is_main:
            wandb_config = {
                "model": type(action_gpt).__name__,
                "lr_max": lr_max,
                "weight_decay": weight_decay,
                "batch_size": bs_per_gpu * accelerator.num_processes,
                "num_epochs": num_epochs,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "act_dim": self.act_dim,
                "sequence_length": self.sequence_length,
                "action_chunk_size": self.action_chunk_size,
                "prev_action_buffer_size": self.prev_action_buffer_size,
                "pred_binary_gripper_action": self.pred_binary_gripper_action
            }
            
            run = wandb.init(
                project="action_gpt", 
                config=wandb_config,
                name=f"actiongpt-run-{time()}",
                reinit=True
            )
            self.wandb_run = run
            
    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def process_index(self):
        return self.accelerator.process_index

    def print(self, *args, **kwargs):
        self.accelerator.print(*args, **kwargs)

    def save_checkpoint(self, save_dir):
        unwrapped_action_gpt = self.accelerator.unwrap_model(self.action_gpt)
        state_dict = unwrapped_action_gpt.get_state_dict_to_save()
        
        torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))
        omegaconf.OmegaConf.save(unwrapped_action_gpt.config, os.path.join(save_dir, "config.yaml"))
        self.print(f"A new model checkpoint is saved to {save_dir}")
        
    def train(self):
        step = 0
        for epoch in range(self.num_epochs+1):
            if epoch != 0:
                self.accelerator.wait_for_everyone()
                save_dir = os.path.join(self.save_path, f'saved_epoch_{epoch}_step_{step}')

                if self.is_main:
                    os.makedirs(save_dir, exist_ok=True)
                    self.save_checkpoint(save_dir)

                if epoch == self.num_epochs:
                    break
                if (self.max_epoch is not None) and (epoch >= self.max_epoch):
                    break

            log_loss = {
                'action_arm': torch.tensor(0).float().to(self.device),
                'action_gripper': torch.tensor(0).float().to(self.device),
            }
            eval_log_loss = {
                'action_arm': torch.tensor(0).float().to(self.device),
                'action_gripper': torch.tensor(0).float().to(self.device),
            }
            
            batch_idx = 0
            batch, load_time = self.train_prefetcher.next()
            while batch is not None:
                with self.accelerator.accumulate(self.action_gpt):
                    self.action_gpt.train()
                    self.optimizer.zero_grad()
                    loss = self.calculate_loss(batch, train=True)
                    self.accelerator.backward(loss['total_loss'])
                    self.optimizer.step()

                    for key in log_loss:
                        if key in loss:
                            log_loss[key] += loss[key].detach() / self.print_steps

                if (batch_idx+1) % self.print_steps == 0:
                    with torch.no_grad():
                        self.action_gpt.eval()
                        batch, _ = self.eval_prefetcher.next_without_none()
                        loss = self.calculate_loss(batch, train=False)
                        for key in eval_log_loss:
                            if key in loss:
                                eval_log_loss[key] = loss[key].detach()

                    self.log(log_loss, eval_log_loss, epoch, batch_idx, step)
                    for key in log_loss:
                        log_loss[key] = torch.tensor(0).float().to(self.device)
                    for key in eval_log_loss:
                        eval_log_loss[key] = torch.tensor(0).float().to(self.device)

                self.scheduler.step()

                # if batch_idx % self.save_steps == 0 and batch_idx > 0:
                #     self.accelerator.wait_for_everyone()
                #     save_dir = os.path.join(self.save_path, f'temp_epoch_{epoch}_step_{step}')

                #     if self.is_main:
                #         existing_ckpt_dirs = glob(os.path.join(self.save_path, f'temp_epoch_*_step_*'))
                #         for existing_ckpt_dir in existing_ckpt_dirs:
                #             if existing_ckpt_dir != save_dir:
                #                 shutil.rmtree(existing_ckpt_dir)
                #         os.makedirs(save_dir, exist_ok=True)
                #         self.save_checkpoint(save_dir)

                batch_idx += 1
                step += 1
                batch, load_time = self.train_prefetcher.next()
                
        if self.is_main:
            wandb.finish()

    def calculate_loss(self, batch, train):
        rgb_initial = self.rgb_preprocessor(batch['rgb_initial'], train=train)
        
        pred = self.action_gpt(
            rgb=rgb_initial[:, 0],  # (b, c, h, w)
            language=batch['lang_input_ids'],
            prev_actions=batch['prev_actions'],
            train=train,
            lang_attention_mask=batch['lang_attention_mask'],
        )
    
        loss = {}
        device = batch['rgb_initial'].device
        
        if self.pred_discrete_arm_action:
            action_arm_loss_func = cross_entropy
            gt_action_arm = batch['actions'][..., :self.act_dim-1].long()
        else:
            action_arm_loss_func = F.smooth_l1_loss
            gt_action_arm = batch['actions'][..., :self.act_dim-1]
        
        loss['action_arm'] = masked_loss(pred['arm_action_preds'], gt_action_arm, batch['mask'], 0, action_arm_loss_func) if pred['arm_action_preds'] is not None else torch.tensor(0.0).to(device)
        
        if self.pred_binary_gripper_action:
            gripper_action_loss_func = F.binary_cross_entropy_with_logits
        else:
            gripper_action_loss_func = F.smooth_l1_loss
            
        loss['action_gripper'] = masked_loss(pred['gripper_action_preds'], batch['actions'][..., -1:].float(), batch['mask'], 0, gripper_action_loss_func) if pred['gripper_action_preds'] is not None else torch.tensor(0.0).to(device)
        total_loss = 100 * loss['action_arm'] + loss['action_gripper']
        loss['total_loss'] = total_loss
        
        return loss

    def log(self, log_loss, eval_log_loss, epoch, batch_idx, step):
        for key in log_loss:
            log_loss[key] = self.accelerator.gather_for_metrics(log_loss[key]).mean()
        for key in eval_log_loss:
            eval_log_loss[key] = self.accelerator.gather_for_metrics(eval_log_loss[key]).mean()

        text = 'Train Epoch: {} [{}/{} ({:.0f}%)] LR:{}'.format(
            epoch, 
            batch_idx * self.bs_per_gpu * self.accelerator.num_processes, 
            len(self.train_prefetcher), 
            100. * batch_idx * self.bs_per_gpu * self.accelerator.num_processes / len(self.train_prefetcher),
            self.scheduler.get_last_lr()[0],
        )
        for key in log_loss:
            text = text + ' {}_loss: {:.5f}'.format(key, log_loss[key])
        for key in eval_log_loss:
            text = text + ' eval_{}_loss: {:.5f}'.format(key, eval_log_loss[key])
        self.print(text)
        
        if self.is_main:
            try:
                wandb_log = {
                    f"{key}_loss": log_loss[key].item() for key in log_loss
                }
                wandb_log.update({
                    f"eval_{key}_loss": eval_log_loss[key].item() for key in eval_log_loss
                })
                wandb_log.update({
                    "learning_rate": self.scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "step": step
                })
                wandb.log(wandb_log, step=step)
                
            except Exception as e:
                self.print(f"Warning: Error in logging: {e}")
                import traceback
                traceback.print_exc()