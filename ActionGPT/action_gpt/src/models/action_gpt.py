import torch
import torch.nn as nn
import torch.nn.functional as F

class ActionGPT(nn.Module):
    def __init__(
            self,
            model_lang,
            model_vision,
            model_causal_transformer,
            act_dim,
            hidden_size,
            sequence_length,
            chunk_size,
            prev_action_buffer_size,
            img_feat_dim,
            patch_feat_dim,
            lang_feat_dim,
            freeze_lang=True,
            freeze_vision=True,
            pred_discrete_arm_action=False, # predict discrete arm actions for berkeley_fanuc_manipulation
            **kwargs
    ):
        super().__init__()
        self.act_dim = act_dim
        self.sequence_length = sequence_length
        self.chunk_size = chunk_size
        self.prev_action_buffer_size = prev_action_buffer_size
        
        # GPT-style backbone
        self.hidden_size = hidden_size
        self.model_causal_transformer = model_causal_transformer

        # Language Encoder (T5-base)
        self.model_lang = model_lang
        self.freeze_lang = freeze_lang
        if freeze_lang:
            for _, param in self.model_lang.named_parameters():
                param.requires_grad = False
        
        # Vision Encoder (SigLIP)
        self.model_vision = model_vision
        self.freeze_vision = freeze_vision
        if freeze_vision:
            for _, param in self.model_vision.named_parameters():
                param.requires_grad = False
                
        self.lang_feat_dim = lang_feat_dim
        self.img_feat_dim = img_feat_dim
        self.patch_feat_dim = patch_feat_dim
        
        # Condition embedding
        self.embed_condition = nn.Embedding(1, hidden_size)
        
        # Embedding function for languages
        self.embed_lang = torch.nn.Linear(self.lang_feat_dim, hidden_size)
        
        # Embedding function for vision
        self.embed_img = torch.nn.Linear(self.img_feat_dim, hidden_size)
        self.embed_patch = torch.nn.Linear(self.patch_feat_dim, hidden_size)
        
        # Embedding functions for previous actions
        self.embed_prev_action = nn.Linear(act_dim, hidden_size)
        
        # Action query tokens (learnable parameters)
        self.action_queries = nn.Embedding(1, hidden_size)  # arm + gripper
        self.action_chunk_queries = nn.Embedding(sequence_length*chunk_size, hidden_size)
        self.action_chunk_queries.weight.data.fill_(0)
        
        # Layer norm
        self.embed_ln = nn.LayerNorm(hidden_size)

        # Cross-attention for vision features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )

        # Action prediction head
        self.pred_act_mlps = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size//2),
            nn.Linear(hidden_size//2, hidden_size//2)
        ])
        
        self.pred_discrete_arm_action = pred_discrete_arm_action
        if self.pred_discrete_arm_action:
            self.pred_arm_act = nn.Linear(hidden_size//2, (self.act_dim-1)*3) # discrete arm action, [0:-0.01, 1:0, 2:+0.01]
        else:
            self.pred_arm_act = nn.Linear(hidden_size//2, self.act_dim-1) # arm action

        self.pred_gripper_act = nn.Linear(hidden_size//2, 1) # gripper action
        
    @property
    def device(self):
        return next(self.parameters()).device

    def get_state_dict_to_save(self):
        modules_to_exclude = ['model_lang', 'model_vision']
        state_dict = {k: v for k, v in self.state_dict().items() if
                      not any(module_name in k for module_name in modules_to_exclude)}
        return state_dict

    def forward(self, 
                rgb,                  # (b, c, h, w)
                language,             # Tokenized language input
                attention_mask=None,  # For transformer attention
                prev_actions=None,    # (b, prev_action_buffer_size, act_dim)
                lang_attention_mask=None,
                **kwargs
    ):
        batch_size = rgb.shape[0]
        
        # Embed language with T5-base
        if self.freeze_lang:
            with torch.no_grad():
                lang_embeddings = self.model_lang(input_ids=language, attention_mask=lang_attention_mask).last_hidden_state
        else:
            lang_embeddings = self.model_lang(input_ids=language, attention_mask=lang_attention_mask).last_hidden_state
        lang_embeddings = self.embed_lang(lang_embeddings.float())  # (b, n_lang_tokens, h)

        # Get vision features from SigLIP
        if self.freeze_vision:
            with torch.no_grad():
                obs_embeddings, patch_embeddings = self.model_vision(rgb)  # (b, 197, h) - CLS token + patch tokens
        else:
            obs_embeddings, patch_embeddings = self.model_vision(rgb)  # (b, 197, h)
        obs_embeddings = self.embed_img(obs_embeddings.float())  # (b, 1, h)
        patch_embeddings = self.embed_patch(patch_embeddings.float())  # (b, 196, h)
       
        # Add conditional embeddings
        condition_embeddings = self.embed_condition.weight.view(1, 1, self.hidden_size)  # (1, 1, h)
        lang_embeddings = lang_embeddings + condition_embeddings  # (b, n_lang_tokens, h)
        patch_embeddings = patch_embeddings + condition_embeddings  # (b, 196, h)
        obs_embeddings = obs_embeddings + condition_embeddings  # (b, 1, h)
        cond_stacked_inputs = torch.cat((lang_embeddings, patch_embeddings, obs_embeddings), dim=1)  # (b, n_cond_tokens, h)
        
        # Embed previous actions if provided
        if prev_actions is not None:
            prev_action_embeddings = self.embed_prev_action(prev_actions)  # (b, prev_action_buffer_size, h)
            prev_action_embeddings = prev_action_embeddings + condition_embeddings  # Add condition embedding
        else:
            # If no previous actions, use zero tensor
            prev_action_embeddings = torch.zeros(batch_size, self.prev_action_buffer_size, self.hidden_size, device=rgb.device)
        
        # Generate action query tokens
        action_chunk_queries = self.action_queries.weight + self.action_chunk_queries.weight
        action_chunk_queries = action_chunk_queries.view(1, self.sequence_length*self.chunk_size, self.hidden_size)
        action_chunk_queries = action_chunk_queries.expand(batch_size, -1, -1)  # (b, sequence_length*chunk_size, h)
        
        # Concatenate language, previous actions, and action queries
        n_lang_tokens = lang_embeddings.shape[1]
        n_patch_tokens = patch_embeddings.shape[1]
        n_obs_tokens = obs_embeddings.shape[1]
        n_cond_tokens = n_lang_tokens + n_patch_tokens + n_obs_tokens
        
        stacked_inputs = torch.cat([
            cond_stacked_inputs,         # (b, n_cond_tokens, h)
            prev_action_embeddings,      # (b, prev_action_buffer_size, h)
            action_chunk_queries         # (b, sequence_length*chunk_size, h)
        ], dim=1)  # (b, n_lang_tokens + prev_action_buffer_size + sequence_length*chunk_size, h)
        
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Prepare causal attention mask if not provided
        if attention_mask is None:
            sequence_length_total = stacked_inputs.shape[1]
            attention_mask = torch.ones((batch_size, 1, sequence_length_total, sequence_length_total), device=rgb.device)
            
            # Apply language attention mask if provided
            if lang_attention_mask is not None:
                lang_mask = lang_attention_mask.unsqueeze(1).unsqueeze(3)  # (b, 1, n_lang_tokens, 1)
                lang_mask = lang_mask.expand(-1, -1, -1, sequence_length_total)  # (b, 1, n_lang_tokens, sequence_length_total)
                attention_mask[:, :, :n_lang_tokens, :] = lang_mask
        
        # GPT forward pass
        transformer_outputs = self.model_causal_transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=attention_mask,
        )
        
        # Get hidden states
        hidden_states = transformer_outputs['last_hidden_state']
        
        # Extract action query hidden states
        action_query_start = n_cond_tokens + self.prev_action_buffer_size
        action_query_end = action_query_start + self.sequence_length * self.chunk_size
        action_query_hidden = hidden_states[:, action_query_start:action_query_end]  # (b, sequence_length*chunk_size, h)
        
        # Cross attention with patch features
        attended_img_features, _ = self.cross_attention(
            query=action_query_hidden,
            key=patch_embeddings,
            value=patch_embeddings
        )
        
        # Combine with original query (residual connection)
        action_embedding = attended_img_features + action_query_hidden
        
        # Action prediction
        for pred_act_mlp in self.pred_act_mlps:
            action_embedding = F.relu(pred_act_mlp(action_embedding))
        
        # Predict arm and gripper actions
        if self.pred_discrete_arm_action:
            arm_action_preds = self.pred_arm_act(action_embedding)  # (b, sequence_length*chunk_size, (act_dim-1)*3)
            arm_action_preds = arm_action_preds.reshape(batch_size, self.sequence_length, self.chunk_size, self.act_dim-1, 3)
        else:
            arm_action_preds = self.pred_arm_act(action_embedding)  # (b, sequence_length*chunk_size, act_dim-1)
            arm_action_preds = arm_action_preds.reshape(batch_size, self.sequence_length, self.chunk_size, self.act_dim-1)
            
        gripper_action_preds = self.pred_gripper_act(action_embedding)  # (b, sequence_length*chunk_size, 1)
        gripper_action_preds = gripper_action_preds.reshape(batch_size, self.sequence_length, self.chunk_size, 1)
            
        prediction = {
            'arm_action_preds': arm_action_preds,
            'gripper_action_preds': gripper_action_preds
        }
            
        return prediction