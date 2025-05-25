import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from common.processors.preprocessor_utils import get_rgb_preprocessor
import torch.nn.functional as F

class ActionGPT_PolicyWrapper:
    def __init__(
            self,
            policy,
            variant,
            lang_tokenizer
    ):
        """Constructor."""
        self.test_chunk_size = variant['test_chunk_size']
        self.is_gripper_binary = variant['is_gripper_binary']
        self.pred_discrete_arm_action = variant['pred_discrete_arm_action']
        self.lang_tokenizer = lang_tokenizer

        # Preprocess
        # input_size = variant['rgb_shape'] 
        input_size = [224, 224]
        rgb_mean = variant['rgb_mean']
        rgb_std = variant['rgb_std']
        self.transform = T.Compose([
            T.Resize(input_size, interpolation=Image.BICUBIC),
            T.Normalize(rgb_mean, rgb_std)
        ])

        self.rgb_preprocessor = get_rgb_preprocessor(
            model_vision_type="siglip",
            vision_aug_config={
                "do_random_resized_crop": False,
                "do_random_shift": False  # 평가 시에는 augmentation 비활성화
            }
        )

        self.policy = policy
        
        self.act_dim = variant['act_dim']
        self.seq_len = variant['seq_len']
        self.chunk_size = variant['chunk_size']
        
        # Previous action buffer
        self.prev_action_buffer_size = variant['prev_action_buffer_size']  # variant.get('prev_action_buffer_size', 10)
        
        # Optional temporal smoothing
        self.use_temporal_ensemble = variant['use_temporal_ensemble']  # variant.get('use_temporal_ensemble', False)
        
    @property
    def device(self):
        return self.policy.device


    def rgb_process(self, rgb):
        """올바른 이미지 전처리"""
        # CALVIN 환경의 (H, W, C) numpy -> (C, H, W) tensor 변환
        if isinstance(rgb, np.ndarray):
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        
        # uint8 -> float 변환
        if rgb.dtype == torch.uint8:
            rgb = rgb.float()
        
        # 0-255 범위를 0-1로 정규화
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        
        # 224x224로 리사이즈 (SigLIP 요구사항)
        rgb = F.interpolate(rgb.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
        rgb = rgb.squeeze(0)  # 배치 차원 제거
        
        # SigLIP은 ImageNet 정규화 사용
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        rgb = (rgb - mean) / std
        
        return rgb

        # rgb = Image.fromarray(rgb)
        # rgb = T.ToTensor()(rgb.convert('RGB'))
        # rgb = self.transform(rgb)
        # return rgb
        
    def reset(self):
        """Reset function."""
        self.rollout_step_counter = 0
        
        # 이전 예측 액션을 저장하는 버퍼 초기화
        self.prev_action_buffer = torch.zeros(1, self.prev_action_buffer_size, self.act_dim)

        if self.use_temporal_ensemble:
            self.action_buffer = np.zeros((self.test_chunk_size, self.act_dim))
            self.action_buffer_mask = np.zeros(self.test_chunk_size, dtype=bool)

    def step(self, obs, goal):
        """Step function."""
        # Language
        lang_inputs = self.lang_tokenizer(goal, return_tensors='pt', padding=True)
        tokenized_text = lang_inputs.input_ids
        lang_attention_mask = lang_inputs.attention_mask

        # RGB
        rgb = self.rgb_process(obs['rgb_obs']['rgb_static'])
        
        # 디버깅: 형태 확인
        # print(f"RGB after processing: {rgb.shape}")
        
        # Forward pass
        tokenized_text = tokenized_text.to(self.device)
        lang_attention_mask = lang_attention_mask.to(self.device) if lang_attention_mask is not None else None
        
        # RGB를 배치 차원 1로 고정
        if rgb.dim() == 3:  # (C, H, W)
            rgb = rgb.unsqueeze(0)  # (1, C, H, W)
        elif rgb.dim() == 4 and rgb.shape[0] != 1:  # 배치 크기가 1이 아닌 경우
            rgb = rgb[:1]  # 첫 번째만 사용하여 배치 크기를 1로 만듦
        
        rgb = rgb.to(self.device)
        prev_actions = self.prev_action_buffer.to(self.device)

        with torch.no_grad():
            prediction = self.policy(
                rgb=rgb, 
                language=tokenized_text,
                prev_actions=prev_actions,
                train=False,
                lang_attention_mask=lang_attention_mask,
            )

        # 예측된 액션 가져오기 (1, t, chunk_size, act_dim)
        arm_action_preds = prediction['arm_action_preds']  # (1, t, chunk_size, act_dim - 1)
        if self.pred_discrete_arm_action:
            arm_action_preds = arm_action_preds.view(-1, self.act_dim - 1, 3)
        else:
            arm_action_preds = arm_action_preds.view(-1, self.act_dim - 1)
        
        # Gripper action
        gripper_action_preds = prediction['gripper_action_preds']  # (1, t, chunk_size, 1)
        gripper_action_preds = gripper_action_preds.view(-1, 1)  # (t*chunk_size, 1)
        
        # Use the first test_chunk_size action
        arm_action_pred = arm_action_preds[:self.test_chunk_size]  # (test_chunk_size, act_dim - 1)
        gripper_action_pred = gripper_action_preds[:self.test_chunk_size]  # (test_chunk_size, 1)
        
        if not self.use_temporal_ensemble:
            if self.is_gripper_binary:
                gripper_action_pred = ((gripper_action_pred > 0).float()) * 2.0 - 1.0
            
            if self.pred_discrete_arm_action:
                arm_action_pred = arm_action_pred.softmax(dim=-1).argmax(dim=-1)
                
            action_pred = torch.cat((arm_action_pred, gripper_action_pred), dim=-1)  # (test_chunk_size, act_dim)
            action_pred = action_pred.detach().cpu()
        else:
            # Shift action buffer
            self.action_buffer[1:, :, :] = self.action_buffer[:-1, :, :]
            self.action_buffer_mask[1:, :] = self.action_buffer_mask[:-1, :]
            self.action_buffer[:, :-1, :] = self.action_buffer[:, 1:, :]
            self.action_buffer_mask[:, :-1] = self.action_buffer_mask[:, 1:]
            self.action_buffer_mask = self.action_buffer_mask * np.fliplr(np.triu(np.ones(self.test_chunk_size))).astype(bool)

            # Add to action buffer
            if self.pred_discrete_arm_action:
                action = torch.cat((arm_action_pred.reshape(arm_action_pred.shape[0], -1), gripper_action_pred), dim=-1) # (t*chunk_size, (act_dim-1)*3+1)
            else:
                action = torch.cat((arm_action_pred, gripper_action_pred), dim=-1) # (t*chunk_size, act_dim)
            action = action.detach().cpu().numpy()
            self.action_buffer[0] = action
            self.action_buffer_mask[0] = True
            
            # Ensemble temporally to predict action
            action_pred = np.sum(self.action_buffer[:, 0, :] * self.action_buffer_mask[:, 0:1], axis=0) / np.sum(self.action_buffer_mask[:, 0], axis=0)
            action_pred = torch.from_numpy(action_pred)

            # Make gripper action either -1 or 1
            if self.is_gripper_binary:
                action_pred[-1] = 1 if action_pred[-1] > 0 else -1
            
            if self.pred_discrete_arm_action:
                arm_action_pred = action_pred[:-1]
                arm_action_pred = arm_action_pred.reshape(-1, 3)
                arm_action_pred = arm_action_pred.softmax(dim=-1).argmax(dim=-1)
                action_pred = torch.cat([arm_action_pred, action_pred[-1:]], dim=-1)
            
            action_pred = action_pred.reshape(1, self.act_dim)

        # 예측된 모든 액션을 flatten하여 버퍼에 추가
        flattened_actions = action_pred[0].reshape(-1, self.act_dim).detach().cpu()
        
        # 버퍼 업데이트: 가장 오래된 액션을 제거하고 새 액션들을 추가
        total_new_actions = flattened_actions.shape[0]
        if total_new_actions >= self.prev_action_buffer_size:
            # 새 액션이 버퍼보다 많거나 같으면 가장 최근 액션만 저장
            self.prev_action_buffer = flattened_actions[-self.prev_action_buffer_size:].unsqueeze(0)
        else:
            # 버퍼의 오래된 액션을 제거하고 새 액션 추가
            self.prev_action_buffer = torch.cat([
                self.prev_action_buffer[:, total_new_actions:], 
                flattened_actions.unsqueeze(0)
            ], dim=1)
        
        self.rollout_step_counter += 1
        return action_pred