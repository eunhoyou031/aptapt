import torch.nn as nn
from transformers import AutoModel

class SigLIPVisionEncoder(nn.Module):
    def __init__(self, pretrained_model_name_or_path):
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        
    def forward(self, images):
        # outputs = self.model.get_image_features(pixel_values=images)  # (b, 1, 768)
        outputs = self.model.vision_model(pixel_values=images).last_hidden_state  # (b, 197, 768)
        obs_embeddings = outputs[:, 0:1, :]  # (b, 1, 768)
        patch_embeddings = outputs[:, 1:, :]  # (b, 196, 768)
        
        return obs_embeddings, patch_embeddings