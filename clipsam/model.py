import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from typing import Optional, List, Tuple, Dict, Any
import einops
from peft import LoraConfig, get_peft_model
from transformers import SamProcessor, SamModel, SiglipModel, SiglipProcessor

# Gradient Reversal Layer implementation
class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from Domain-Adversarial Training paper.
    Forward pass: identity function
    Backward pass: multiply gradients by -lambda (scale factor)
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_
        return output, None

class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer wrapper"""
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_):
        self.lambda_ = lambda_

class DomainClassifier(nn.Module):
    """
    Domain classifier MLP for domain adversarial training.
    Takes SigLIP pooled visual features and predicts domain (iSAID, DeepGlobe, LoveDA).
    """
    def __init__(self, input_dim, num_domains=3, hidden_dim=512, dropout=0.5):
        super(DomainClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_domains)
        )
    
    def forward(self, x):
        return self.classifier(x)

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 norm_cfg=None, act_cfg=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        
        # Normalization layer
        self.norm = None
        if norm_cfg is not None:
            if norm_cfg.get('type') == 'SyncBN':
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm_cfg.get('type') == 'BN':
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm_cfg.get('type') == 'LN':
                self.norm = nn.LayerNorm(out_channels)
        
        # Activation layer
        self.act = None
        if act_cfg is not None:
            if act_cfg.get('type') == 'ReLU':
                self.act = nn.ReLU(inplace=True)
            elif act_cfg.get('type') == 'GELU':
                self.act = nn.GELU()
            elif act_cfg.get('type') == 'LeakyReLU':
                self.act = nn.LeakyReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x

class SigLipSamSegmentator(nn.Module):
    def __init__(self, 
                 siglip_model_name='google/siglip2-so400m-patch14-384',
                 sam_model_name='facebook/sam-vit-base',
                 down_spatial_times=2,
                 with_dense_feat=True,
                 lora_cfg=None,
                 device=None,
                 enable_domain_adaptation=False,
                 num_domains=3,
                 domain_classifier_lr_scale=1.0):
        super().__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Domain adaptation settings
        self.enable_domain_adaptation = enable_domain_adaptation
        self.num_domains = num_domains
        self.domain_classifier_lr_scale = domain_classifier_lr_scale
            
        # Add LoRA configuration with the exact params from the original implementation
        if lora_cfg is None:
            self.lora_cfg = {
                'backbone': {
                    'r': 16,
                    'lora_alpha': 16,
                    'lora_dropout': 0.1,
                    'target_modules': ['qkv', 'proj', 'lin1', 'lin2', 'neck.conv1', 'neck.conv2']
                },
                'clip_vision_encoder': {
                    'r': 16,
                    'lora_alpha': 16,
                    'lora_dropout': 0.1,
                    'target_modules': ['k_proj', 'v_proj', 'q_proj', 'out_proj']
                },
                'clip_text_encoder': {
                    'r': 16,
                    'lora_alpha': 16,
                    'lora_dropout': 0.1,
                    'target_modules': ['q_proj', 'k_proj', 'v_proj', 'out_proj', 
                                       'fc1', 'fc2', 'dense', 'head']
                }
            }
        else:
            self.lora_cfg = lora_cfg
            
        # Load SigLIP models
        self.clip_vision_processor = SiglipProcessor.from_pretrained(siglip_model_name).image_processor
        self.clip_text_processor = SiglipProcessor.from_pretrained(siglip_model_name).tokenizer
        
        siglip_model = SiglipModel.from_pretrained(siglip_model_name).to(self.device)
        self.clip_vision_encoder = siglip_model.vision_model
        self.clip_text_encoder = siglip_model.text_model
        self.logit_scale = siglip_model.logit_scale
        self.logit_bias = siglip_model.logit_bias
        
        # Load SAM model
        self.sam_processor = SamProcessor.from_pretrained(sam_model_name)
        sam_model = SamModel.from_pretrained(sam_model_name).to(self.device)
        self.backbone = sam_model.vision_encoder
        self.sam_prompt_encoder = sam_model.prompt_encoder
        self.sam_mask_decoder = sam_model.mask_decoder
        self.shared_image_embedding = sam_model.shared_image_embedding
        
        # Config parameters
        self.down_spatial_times = down_spatial_times
        self.with_dense_feat = with_dense_feat
        
        # Define norms and activations
        self.norm_cfg = dict(type='SyncBN', requires_grad=True)
        self.act_cfg = dict(type='GELU')
        
        # Prompter modules
        self.prompter_down_channel = ConvModule(
            in_channels=self.clip_text_encoder.config.hidden_size*3,
            out_channels=self.sam_prompt_encoder.hidden_size,
            kernel_size=1,
            stride=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        # Sequential downsampling blocks
        down_spatial_modules = []
        for _ in range(down_spatial_times):
            down_spatial_modules.append(ConvModule(
                in_channels=self.sam_prompt_encoder.hidden_size,
                out_channels=self.sam_prompt_encoder.hidden_size,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        
        # Final 1x1 conv without normalization or activation
        down_spatial_modules.append(ConvModule(
            in_channels=self.sam_prompt_encoder.hidden_size,
            out_channels=self.sam_prompt_encoder.hidden_size,
            kernel_size=1,
            stride=1,
            norm_cfg=None,
            act_cfg=None))
        
        self.prompter_down_spatial = nn.Sequential(*down_spatial_modules)
        
        # Domain adaptation components
        if self.enable_domain_adaptation:
            # Gradient reversal layer (shared between classifiers)
            self.gradient_reversal = GradientReversalLayer(lambda_=1.0)
            
            # Domain classifier for SigLIP vision pooled features
            siglip_hidden_size = self.clip_text_encoder.config.hidden_size
            self.siglip_domain_classifier = DomainClassifier(
                input_dim=siglip_hidden_size,
                num_domains=self.num_domains,
                hidden_dim=512,
                dropout=0.5
            )
            
            # Domain classifier for SAM vision features
            # SAM features are [B, 256, 64, 64], so we need to pool them first
            self.sam_feature_pool = nn.AdaptiveAvgPool2d((1, 1))
            sam_feature_dim = 256  # As per SAM neck output
            self.sam_domain_classifier = DomainClassifier(
                input_dim=sam_feature_dim,
                num_domains=self.num_domains,
                hidden_dim=256, # Smaller hidden dim for smaller input
                dropout=0.5
            )
            
            print(f"Domain adaptation enabled with {self.num_domains} domains")
            print(f"SigLIP domain classifier input dim: {siglip_hidden_size}")
            print(f"SAM domain classifier input dim: {sam_feature_dim}")
        else:
            self.gradient_reversal = None
            self.siglip_domain_classifier = None
            self.sam_domain_classifier = None
            self.sam_feature_pool = None
        
        # Freeze models by default (can be unfrozen selectively)
        self._freeze_models()
        
        # After all components are initialized, set up fine-tuning parameters
        self.set_finetune_parameters()
        self.print_trainable_parameters()
        
    def _freeze_models(self):
        """Freeze all pretrained models"""
        for param in self.clip_vision_encoder.parameters():
            param.requires_grad = False
        for param in self.clip_text_encoder.parameters():
            param.requires_grad = False
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.sam_prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.sam_mask_decoder.parameters():
            param.requires_grad = False
        for param in self.shared_image_embedding.parameters():
            param.requires_grad = False
        
        # IMPORTANT: The prompter networks should always be trainable
        # They are the key components that learn to connect SigLIP and SAM
        self.prompter_down_channel.requires_grad_(True)
        self.prompter_down_spatial.requires_grad_(True)
        
        # Domain adaptation components should also be trainable if enabled
        if self.enable_domain_adaptation:
            self.siglip_domain_classifier.requires_grad_(True)
            self.sam_domain_classifier.requires_grad_(True)
    
    def unfreeze_components(self, component_list):
        """Selectively unfreeze components for fine-tuning"""
        component_map = {
            'clip_vision_encoder': self.clip_vision_encoder,
            'clip_text_encoder': self.clip_text_encoder,
            'backbone': self.backbone,
            'sam_prompt_encoder': self.sam_prompt_encoder,
            'sam_mask_decoder': self.sam_mask_decoder
        }
        
        for component_name in component_list:
            if component_name in component_map:
                for param in component_map[component_name].parameters():
                    param.requires_grad = True
                print(f"Unfrozen {component_name}")
            else:
                print(f"Warning: {component_name} not found")
    
    def get_image_positional_embeddings(self, size):
        """Get positional embeddings for the image"""
        target_device = self.shared_image_embedding.positional_embedding.device
        target_dtype = self.shared_image_embedding.positional_embedding.dtype
        grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size

        positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)  # channel x height x width

    def extract_features(self, image, text_list, return_domain_logits=False):
        """Extract features from both SigLIP and SAM encoders"""
        # Prepare text inputs
        if isinstance(text_list, str):
            text_list = [text_list]
            
        # For SAM vision encoder - need to unnormalize the image first
        # ImageNet mean/std used in the dataset transform
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
        
        # Unnormalize the image for SAM
        unnormalized_image = image * std + mean
        
        # SAM requires 1024x1024 input resolution
        sam_image = F.interpolate(unnormalized_image, size=(1024, 1024), mode='bilinear', align_corners=False)
        
        # Clip values to [0, 1] range to ensure proper processing
        sam_image = torch.clamp(sam_image, 0, 1)
        
        # Process with SAM's vision encoder
        sam_visual_feat = self.backbone(sam_image)
        sam_visual_feat = sam_visual_feat['last_hidden_state']  # BX256X64X64

        # For SigLIP vision encoder - use the normalized image directly
        siglip_image_size = list(self.clip_vision_processor.size.values())
        x_clip = F.interpolate(image, size=siglip_image_size, mode='bilinear', align_corners=False)
        
        clip_visual_feat = self.clip_vision_encoder(x_clip)
        clip_visual_feat_pooler = clip_visual_feat['pooler_output']  # BX1152
        clip_visual_feat = clip_visual_feat['last_hidden_state']  # BX729X1152
        clip_visual_feat = einops.rearrange(clip_visual_feat, 'b (h w) c -> b c h w', 
                                          h=int(math.sqrt(clip_visual_feat.shape[1])))  # BX1152X27X27
        
        # Domain prediction using gradient reversal (if enabled)
        domain_logits = None
        if self.enable_domain_adaptation and return_domain_logits:
            # 1. SigLIP domain prediction
            siglip_reversed_features = self.gradient_reversal(clip_visual_feat_pooler)
            siglip_domain_logits = self.siglip_domain_classifier(siglip_reversed_features)
            
            # 2. SAM domain prediction
            sam_pooled_feat = self.sam_feature_pool(sam_visual_feat).flatten(1)
            sam_reversed_features = self.gradient_reversal(sam_pooled_feat)
            sam_domain_logits = self.sam_domain_classifier(sam_reversed_features)
            
            domain_logits = {
                'siglip': siglip_domain_logits,
                'sam': sam_domain_logits
            }

        # Process text with SigLIP
        text_dict = self.clip_text_processor(text_list, return_tensors='pt', padding=True, truncation=True, max_length=128)
        text_dict = {k: v.to(image.device) for k, v in text_dict.items()}
        input_ids = text_dict['input_ids']
        clip_text_feat = self.clip_text_encoder(**text_dict)
        clip_text_feat_pooler = clip_text_feat['pooler_output']  # BX1152
        clip_text_feat = clip_text_feat['last_hidden_state']  # BX7X1152

        # Normalize features for cosine similarity
        normalized_clip_visual_feat = clip_visual_feat / clip_visual_feat.norm(p=2, dim=1, keepdim=True)  # BX1152X27X27
        normalized_clip_text_feat = clip_text_feat / clip_text_feat.norm(p=2, dim=2, keepdim=True)  # BX7X1152
        normalized_clip_text_feat_pooler = clip_text_feat_pooler / clip_text_feat_pooler.norm(p=2, dim=1, keepdim=True)  # BX1152

        # Local activation - text token level
        local_activate = einops.einsum(normalized_clip_visual_feat, normalized_clip_text_feat, 'b c h w, b d c -> b d h w')
        local_activate = local_activate * self.logit_scale.exp() + self.logit_bias
        local_activate = F.sigmoid(local_activate)  # BX7X27X27
        local_activated_feat = einops.einsum(local_activate, clip_visual_feat, 'b d h w, b c h w -> b c h w') / local_activate.size(1)  # BX1152X27X27
        local_clip_visual_feat = (clip_visual_feat + local_activated_feat) / 2

        # Global activation - text pooler level
        global_activate = einops.einsum(normalized_clip_visual_feat, normalized_clip_text_feat_pooler, 'b c h w, b c -> b h w')
        global_activate_logit = global_activate * self.logit_scale.exp() + self.logit_bias
        global_activate = F.sigmoid(global_activate_logit)  # BX27X27
        global_activated_feat = einops.einsum(global_activate, clip_visual_feat, 'b h w, b c h w -> b c h w')  # BX1152X27X27

        # Combine features and process through prompter networks
        clip_activated_feat = torch.cat([local_clip_visual_feat, global_activated_feat, clip_visual_feat], dim=1)
        clip_activated_feat = self.prompter_down_channel(clip_activated_feat)
        clip_activated_feat = clip_activated_feat + self.get_image_positional_embeddings(clip_activated_feat.size(2))
        clip_activated_feat = self.prompter_down_spatial(clip_activated_feat)  # BX768X7X7

        # Prepare SAM prompt encoder inputs
        batch_size = image.size(0)
        image_positional_embeddings = self.get_image_positional_embeddings(sam_visual_feat.size(2))
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

        # Process with SAM prompt encoder
        if self.with_dense_feat:
            global_activate_logit = F.interpolate(
                global_activate_logit.unsqueeze(1), 
                size=(image_positional_embeddings.size(2)*4, image_positional_embeddings.size(3)*4), 
                mode='bilinear', 
                align_corners=False
            )
            sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
                input_points=None,
                input_labels=None,
                input_boxes=None,
                input_masks=global_activate_logit,
            )
        else:
            sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
                input_points=None,
                input_labels=None,
                input_boxes=None,
                input_masks=None,
            )

        # Replace sparse embeddings with processed SigLIP features
        sparse_embeddings = einops.rearrange(clip_activated_feat, 'b c h w -> b 1 (h w) c')

        # Generate masks with SAM mask decoder
        low_res_masks, iou_predictions, _ = self.sam_mask_decoder(
            image_embeddings=sam_visual_feat,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            attention_similarity=None,
            target_embedding=None,
            output_attentions=None,
        )
        
        # Get the segmentation mask and ensure correct shape
        # SAM returns masks with shape [B, 1, H, W]
        seg_mask = low_res_masks.squeeze(1)  # Remove the extra dimension, now [B, H, W]
        
        if return_domain_logits:
            return seg_mask, domain_logits
        else:
            return seg_mask
    
    def dice_loss(self, pred, target):
        """Calculate Dice loss for binary segmentation"""
        smooth = 1.0
        
        # Apply sigmoid to convert logits to probabilities
        pred = torch.sigmoid(pred)
        
        # Flatten prediction and target tensors
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        
        return 1 - dice
    
    def compute_loss(self, pred, target, lambda_ce=0.9):
        """Combined loss: Cross-Entropy + Dice Loss"""
        # Resize target to match prediction
        target_resized = F.interpolate(
            target.unsqueeze(1),
            size=pred.shape[1:],
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        
        # Convert to binary
        target_binary = (target_resized > 0.0).float()
        
        # Binary cross entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target_binary)
        
        # Dice loss
        dice = self.dice_loss(pred, target_binary)
        
        # Combined loss
        loss = lambda_ce * bce_loss + (1 - lambda_ce) * dice
        
        return loss
    
    def compute_domain_loss(self, domain_logits, domain_labels, sam_loss_weight=0.5):
        """
        Compute domain classification loss for both SigLIP and SAM classifiers.
        domain_logits: A dictionary containing 'siglip' and 'sam' logits.
        domain_labels: The ground truth domain labels.
        sam_loss_weight: A float to weigh the SAM domain loss.
        """
        if domain_logits is None or domain_labels is None:
            return torch.tensor(0.0, device=self.device)
        
        # SigLIP domain loss
        siglip_loss = F.cross_entropy(domain_logits['siglip'], domain_labels)
        
        # SAM domain loss
        sam_loss = F.cross_entropy(domain_logits['sam'], domain_labels)
        
        # Total domain loss
        total_loss = siglip_loss + (sam_loss * sam_loss_weight)
        
        return total_loss
    
    def set_gradient_reversal_lambda(self, lambda_):
        """Set the lambda parameter for gradient reversal layer"""
        if self.gradient_reversal is not None:
            self.gradient_reversal.set_lambda(lambda_)
    
    def forward(self, image, text_list, domain_labels=None, return_domain_logits_inference=False):
        """Forward pass of the model"""
        # Use memory efficient attention
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        # Determine if we should return domain logits
        # This is true if we are training with domain adaptation OR if it's explicitly requested for inference
        return_domain_logits = (self.enable_domain_adaptation and domain_labels is not None) or \
                               (self.enable_domain_adaptation and return_domain_logits_inference)
        
        if return_domain_logits:
            seg_mask, domain_logits = self.extract_features(image, text_list, return_domain_logits=True)
        else:
            seg_mask = self.extract_features(image, text_list, return_domain_logits=False)
            domain_logits = None
        
        # Ensure seg_mask has correct shape before interpolation [B, H, W]
        if len(seg_mask.shape) == 4:
            # If somehow it's already [B, 1, H, W]
            seg_mask = seg_mask.squeeze(1)
        
        # Resize to input resolution
        input_size = image.shape[2]
        seg_mask = F.interpolate(
            seg_mask.unsqueeze(1),  # Add channel dimension [B, 1, H, W]
            size=(input_size, input_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # Remove channel dimension [B, H, W]
        
        if return_domain_logits:
            return seg_mask, domain_logits
        else:
            return seg_mask
    
    def predict_masks(self, image, text, threshold=0.5):
        """Predict binary masks from image and text"""
        # Forward pass to get mask logits (no domain labels for inference)
        output = self.forward(image, text, domain_labels=None)
        
        # Handle both single return (seg_mask) and tuple return (seg_mask, domain_logits)
        if isinstance(output, tuple):
            mask_logits = output[0]
        else:
            mask_logits = output
        
        # Apply sigmoid and threshold
        mask_probs = torch.sigmoid(mask_logits)
        binary_masks = (mask_probs > threshold).float()
        
        return {
            'logits': mask_logits,
            'probs': mask_probs,
            'masks': binary_masks
        }
    
    def set_finetune_parameters(self):
        """Apply LoRA to specified components and freeze others"""
        # Only apply LoRA to the same components as the original implementation
        peft_keys = ['backbone', 'clip_vision_encoder', 'clip_text_encoder']
        
        for k in peft_keys:
            if k in self.lora_cfg:
                v_ = self.lora_cfg[k].copy()
                lora_config = LoraConfig(**v_)
                # Apply LoRA to the component
                setattr(self, k, get_peft_model(getattr(self, k), lora_config))
                print(f"Applied LoRA to {k}")
            else:
                # If not using LoRA for this component, freeze it
                print(f"Warning: {k} not in lora_cfg")
                getattr(self, k).requires_grad_(False)
                print(f"Set {k} to requires_grad=False")
        
        # Ensure prompt encoder and mask decoder are frozen (not using LoRA)
        self.sam_prompt_encoder.requires_grad_(False)
        self.sam_mask_decoder.requires_grad_(False)
    
    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        """Count trainable parameters vs total parameters"""
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                if hasattr(param, "element_size"):
                    num_bytes = param.element_size()
                elif not hasattr(param, "quant_storage"):
                    num_bytes = 1
                else:
                    num_bytes = param.quant_storage.itemsize
                num_params = num_params * 2 * num_bytes

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    def print_trainable_parameters(self) -> None:
        """Print the percentage of trainable parameters"""
        trainable_params, all_param = self.get_nb_trainable_parameters()

        # Debug prints for each component
        print("\nParameter counts by component:")
        total_trainable = 0
        for name, module in self.named_children():
            # Skip None modules (like gradient_reversal and domain_classifier when domain adaptation is disabled)
            if module is None:
                continue
            total_params = sum(p.numel() for p in module.parameters())
            module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_trainable += module_trainable
            print(f"{name}: {total_params:,} total params, {module_trainable:,} trainable params")

        print(
            f"\ntrainable params: {total_trainable:,d} || all params: {all_param:,d} || trainable%: {100 * total_trainable / all_param:.4f}"
        ) 