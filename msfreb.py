"""
This module defines the main architecture of the Msfreb model, including normalization layers,
the Pyramid Pooling Module (PPM), custom loss functions (FocalLoss, DiceLoss), and the main
model class that integrates a backbone network with feature enhancement and prediction heads.
"""

from timm.layers import trunc_normal_
from modules.decoderhead import AdvancedPredictHead, LayerNorm as CustomDecoderLayerNorm
from modules.convnextv2 import ConvNeXtV2, convnextv2_tiny, convnextv2_base, convnextv2_large
from modules.bifpn import BiFPN
from modules.wavelet_enhancement import ImprovedWaveletEnhancement, MultiScaleWaveletEnhancement
from modules.ppm import PPM
from utils.loss import FocalLoss, DiceLoss
from utils.flexible_norm import FlexibleNorm
from functools import partial
import torch.nn as nn
import torch
import torch.nn.functional as F

import sys
sys.path.append('./modules')

class Msfreb(nn.Module):
    """
    The main Msfreb model for image manipulation detection and localization.
    This model integrates a backbone (e.g., ConvNeXt), a BiFPN for feature fusion,
    optional enhancement modules like PPM and Wavelet Enhancement, and a prediction head.
    """
    def __init__(
        self, 
        input_size = 1024,
        backbone_type = "convnext",
        convnext_variant = "base",
        convnext_pretrain_path = None,
        fpn_channels = 256,
        mlp_embeding_dim = 256, 
        predict_head_norm = "BN",
        predict_head_dropout_rate = 0.1,
        edge_lambda = 20,
        focal_alpha=0.25, 
        focal_gamma=2.0,
        dice_smooth=1.0,
        dice_weight=1.0,
        bifpn_attention = True,
        bifpn_use_p8 = False,
        classification_loss_weight: float = 1.0,
        seg_penalty_fp: float = 1.5,
        seg_penalty_fn: float = 1.5,
        clf_feature_backbone_index: int = 2,
        consistency_loss_weight: float = 0.5,
        use_ppm: bool = True,
        ppm_pool_scales: tuple = (1, 2, 3, 6),
        use_wavelet_enhancement: bool = True,
        wavelet_reduction_ratio: int = 4,
        ppm_norm_type: str = 'gn'
    ):
        """
        Initializes the Msfreb model.

        Args:
            input_size (int): The size of the input image.
            backbone_type (str): The type of backbone network to use.
            convnext_variant (str): The variant of ConvNeXt if it is used as the backbone.
            convnext_pretrain_path (str): Path to the pretrained weights for ConvNeXt.
            fpn_channels (int): Number of channels in the FPN layers.
            mlp_embeding_dim (int): Embedding dimension for the MLP in the prediction head.
            predict_head_norm (str): Normalization type for the prediction head.
            predict_head_dropout_rate (float): Dropout rate in the prediction head.
            edge_lambda (float): Weight for the edge loss component.
            focal_alpha (float): Alpha parameter for the Focal Loss.
            focal_gamma (float): Gamma parameter for the Focal Loss.
            dice_smooth (float): Smoothing factor for the Dice Loss.
            dice_weight (float): Weight for the Dice Loss component.
            bifpn_attention (bool): Whether to use attention in the BiFPN.
            bifpn_use_p8 (bool): Whether to use the P8 feature level in the BiFPN.
            classification_loss_weight (float): Weight for the image-level classification loss.
            seg_penalty_fp (float): Penalty factor for false positive segmentation predictions.
            seg_penalty_fn (float): Penalty factor for false negative segmentation predictions.
            clf_feature_backbone_index (int): Index of the backbone feature to use for classification.
            consistency_loss_weight (float): Weight for the consistency loss.
            use_ppm (bool): Flag to enable or disable the PPM module.
            ppm_pool_scales (tuple): Pool scales for the PPM module.
            use_wavelet_enhancement (bool): Flag to enable or disable the wavelet enhancement module.
            wavelet_reduction_ratio (int): Channel reduction ratio for the wavelet enhancement module.
            ppm_norm_type (str): Normalization type for the PPM module.
        """
        super(Msfreb, self).__init__()
        self.input_size = input_size
        self.backbone_type = backbone_type
        self.dice_weight = dice_weight
        self.edge_lambda = edge_lambda
        self.classification_loss_weight = classification_loss_weight
        self.seg_penalty_fp = seg_penalty_fp
        self.seg_penalty_fn = seg_penalty_fn
        self.clf_feature_backbone_index = clf_feature_backbone_index
        self.consistency_loss_weight = consistency_loss_weight
        self.use_ppm = use_ppm
        self.use_wavelet_enhancement = use_wavelet_enhancement
        self.ppm_norm_type = ppm_norm_type

        if backbone_type == "convnext":
            if convnext_variant == "tiny":
                depths, dims = [3, 3, 9, 3], [96, 192, 384, 768] 
            elif convnext_variant == "base":
                depths, dims = [3, 3, 27, 3], [128, 256, 512, 1024]
            elif convnext_variant == "large":
                depths, dims = [3, 3, 27, 3], [192, 384, 768, 1536]
            else:
                raise ValueError(f"Unsupported ConvNeXt variant: {convnext_variant}")
            
            self.dims_from_backbone = dims
            convnext_out_indices = (0, 1, 2, 3) # Corresponds to C2, C3, C4, C5 features
            self.encoder_net = globals()[f"convnextv2_{convnext_variant}"](
                in_chans=3, 
                num_classes=1000, # Standard ImageNet classes, will be replaced.
                out_indices=convnext_out_indices
            )
            if hasattr(self.encoder_net, 'head'):
                self.encoder_net.head = nn.Identity() # Remove the final classification head.
            
            self.convnext_pretrain_path = convnext_pretrain_path
            self.c2_skip_channels = dims[0]
            bifpn_input_conv_channels = dims[1:4] # C3, C4, C5 channels for BiFPN
        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")
        
        # Initialize PPM if enabled, applied to the deepest feature layer (C5).
        if self.use_ppm:
            ppm_in_channels = self.dims_from_backbone[3]
            self.ppm_module = PPM(
                in_channels=ppm_in_channels,
                out_channels=fpn_channels, 
                pool_scales=ppm_pool_scales,
                norm_type=self.ppm_norm_type,
                act_layer=nn.ReLU
            )
            # Downsample layer for PPM input features.
            self.c5_downsample = nn.Sequential(
                nn.Conv2d(ppm_in_channels, ppm_in_channels, 3, stride=2, padding=1),
                FlexibleNorm(ppm_in_channels, norm_type=self.ppm_norm_type),
                nn.ReLU(inplace=True),
                nn.Conv2d(ppm_in_channels, ppm_in_channels, 3, stride=2, padding=1),
                FlexibleNorm(ppm_in_channels, norm_type=self.ppm_norm_type),
                nn.ReLU(inplace=True)
            )
        else:
            self.ppm_module = None

        # 1x1 convolution to revert PPM output channels back to C5's original channel count.
        if self.use_ppm:
            self.ppm_to_c5_channels_conv = nn.Conv2d(fpn_channels, self.dims_from_backbone[3], kernel_size=1)
        else:
            self.ppm_to_c5_channels_conv = None

        # Initialize Wavelet-like Feature Enhancement for backbone features if enabled.
        if self.use_wavelet_enhancement:
            # Create enhancement modules for C3, C4, and C5 feature levels.
            self.wavelet_c3 = ImprovedWaveletEnhancement(bifpn_input_conv_channels[0], level_idx=0)
            self.wavelet_c4 = ImprovedWaveletEnhancement(bifpn_input_conv_channels[1], level_idx=1) 
            self.wavelet_c5 = ImprovedWaveletEnhancement(bifpn_input_conv_channels[2], level_idx=2)
        else:
            self.wavelet_c3 = None
            self.wavelet_c4 = None
            self.wavelet_c5 = None

        self.featurePyramid_net = BiFPN(
            num_channels=fpn_channels,
            conv_channels=bifpn_input_conv_channels,
            first_time=True, attention=bifpn_attention, use_p8=bifpn_use_p8
        )
        
        self.predict_head = AdvancedPredictHead(
            bifpn_output_channels=fpn_channels,
            c2_channels=self.c2_skip_channels,
            num_bifpn_features=5, # P3, P4, P5, P6, P7
            decoder_embed_dim=mlp_embeding_dim, # This is for the segmentation decoder's internal embedding
            predict_channels=1, # Assuming binary segmentation
            norm_type=predict_head_norm,
            dropout_rate=predict_head_dropout_rate
        )
            
        # Classification Head
        # Input dimension now comes from the selected backbone feature layer
        if not (0 <= self.clf_feature_backbone_index < len(self.dims_from_backbone)):
            raise ValueError(f"Invalid clf_feature_backbone_index: {self.clf_feature_backbone_index}. "
                             f"Must be between 0 and {len(self.dims_from_backbone) - 1}.")
        self.classification_head_input_dim = self.dims_from_backbone[self.clf_feature_backbone_index]
        
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    # (N, C_clf, 1, 1)
            nn.Flatten(),               # (N, C_clf)
            nn.Linear(self.classification_head_input_dim, self.classification_head_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(predict_head_dropout_rate),
            nn.Linear(self.classification_head_input_dim // 2, 1) # Output single logit for classification
        )
            
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.classification_loss_fn = nn.BCEWithLogitsLoss()
        # New: Consistency loss function
        if self.consistency_loss_weight > 0:
            self.consistency_loss_fn = nn.BCEWithLogitsLoss()

        self.apply(self._init_weights)
        
        if backbone_type == "convnext" and convnext_pretrain_path is not None:
            self._load_convnext_weights()
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.InstanceNorm2d)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            
    def _load_convnext_weights(self):
        if self.convnext_pretrain_path and self.backbone_type == "convnext":
            state_dict = torch.load(self.convnext_pretrain_path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            
            # Adjust for out_indices if pretrain model has different structure (e.g. no separate features)
            # This basic loading assumes the encoder_net structure is compatible enough.
            missing_keys, unexpected_keys = self.encoder_net.load_state_dict(state_dict, strict=False)
            print(f"Loaded ConvNeXt backbone weights from '{self.convnext_pretrain_path}'.")
            if missing_keys: print(f"Missing keys: {missing_keys}")
            if unexpected_keys: print(f"Unexpected keys: {unexpected_keys}")
        
    def forward(self, x: torch.Tensor, masks: torch.Tensor, edge_masks: torch.Tensor, shape=None):
        """
        Forward pass of the Msfreb model.

        The process includes:
        1.  Backbone feature extraction (C2, C3, C4, C5).
        2.  Optional feature enhancement with Wavelet and PPM modules.
        3.  Multi-scale feature fusion using BiFPN.
        4.  Segmentation mask prediction using the AdvancedPredictHead.
        5.  Image-level classification using a dedicated head.
        6.  Calculation of a composite loss, including segmentation, edge, classification,
            and consistency losses with penalty logic.

        Args:
            x (torch.Tensor): Input images, shape (N, 3, H, W).
            masks (torch.Tensor): Ground truth segmentation masks, shape (N, 1, H, W).
            edge_masks (torch.Tensor): Ground truth edge masks, shape (N, 1, H, W).
            shape (optional): Not currently used, for future compatibility.

        Returns:
            tuple: A tuple containing:
                - total_loss (torch.Tensor): The combined loss for optimization.
                - mask_pred_prob (torch.Tensor): The predicted segmentation probability map.
                - tamper_pred_prob (torch.Tensor): The predicted image-level tamper probability.
                - final_focal_loss (torch.Tensor): The calculated focal loss component.
                - edge_loss_val (torch.Tensor): The calculated edge loss component.
                - final_dice_loss (torch.Tensor): The calculated dice loss component.
                - classification_loss (torch.Tensor): The calculated classification loss component.
                - consistency_loss_val (torch.Tensor): The calculated consistency loss component.
        """
        # --- 0. Input Preparation ---
        # Ensure input tensors are of type float32 to prevent dtype errors.
        x = x.float()
        masks = masks.float()
        edge_masks = edge_masks.float()
        
        # --- 1. Backbone Feature Extraction ---
        # The encoder network returns a list of feature maps: [C2, C3, C4, C5].
        backbone_features = self.encoder_net(x) 
        
        c2_skip_feature = backbone_features[0]      # C2 is used as a skip connection for the decoder.
        bifpn_input_features = backbone_features[1:] # C3, C4, C5 are inputs to the BiFPN.
        # Select the feature map specified by the index for the classification head.
        features_for_clf_head = backbone_features[self.clf_feature_backbone_index]

        # --- 2. Optional Feature Enhancement ---
        # Apply enhancement modules before the BiFPN.
        if self.use_wavelet_enhancement:
            c3_enhanced = self.wavelet_c3(bifpn_input_features[0])
            c4_enhanced = self.wavelet_c4(bifpn_input_features[1])
            c5_enhanced = self.wavelet_c5(bifpn_input_features[2])
            bifpn_input_features = [c3_enhanced, c4_enhanced, c5_enhanced]

        if self.use_ppm and self.ppm_module is not None:
            # Apply PPM to a downsampled version of the C5 feature for wider context.
            c5_deep = self.c5_downsample(bifpn_input_features[2])
            ppm_enhanced_deep = self.ppm_module(c5_deep)
            # Upsample the PPM output back to C5's resolution.
            ppm_enhanced_c5 = F.interpolate(ppm_enhanced_deep, 
                                          size=bifpn_input_features[2].shape[2:], 
                                          mode='bilinear', align_corners=False)
            # Align channels and add back to the original C5 feature.
            ppm_c5_aligned = self.ppm_to_c5_channels_conv(ppm_enhanced_c5)
            bifpn_input_features[2] = bifpn_input_features[2] + ppm_c5_aligned

        # --- 3. BiFPN for Multi-Scale Feature Fusion ---
        # The BiFPN processes [C3, C4, C5] and outputs a tuple of fused features (P3, P4, P5, P6, P7).
        fpn_outputs_tuple = self.featurePyramid_net(bifpn_input_features) 
        fpn_output_features = list(fpn_outputs_tuple)
        
        # --- 4. Segmentation Head ---
        # The decoder predicts low-resolution logits from BiFPN features and the C2 skip connection.
        logits_low_res, _ = self.predict_head(fpn_output_features, c2_skip_feature)
        
        # Upsample logits to the original input size for final prediction and loss calculation.
        mask_pred_logits = F.interpolate(logits_low_res, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        mask_pred_prob = torch.sigmoid(mask_pred_logits)

        # --- 5. Image-Level Classification Head ---
        # Predict a single tamper logit for the entire image.
        tamper_logits = self.classification_head(features_for_clf_head) # Shape: (N, 1)
        tamper_pred_prob = torch.sigmoid(tamper_logits)

        # --- 6. Loss Calculation ---
        # Generate image-level ground truth: 1 if tampered (mask has non-zero pixels), 0 otherwise.
        is_tampered_gt = (masks.sum(dim=[1, 2, 3]) > 1e-5).float().unsqueeze(1) # Shape: (N, 1)

        # --- 6.1. Supervised Classification Loss ---
        classification_loss = self.classification_loss_fn(tamper_logits, is_tampered_gt) * self.classification_loss_weight

        # --- 6.2. Segmentation Losses (calculated per-sample for penalty logic) ---
        focal_loss_ps = self.focal_loss(mask_pred_logits, masks, reduction='per_sample')  # Shape: (N,)
        dice_loss_ps = self.dice_loss(mask_pred_logits, masks, reduction_per_sample=True)  # Shape: (N,)

        # Edge loss is calculated on the mean and is not subject to per-sample penalties.
        edge_bce_map = F.binary_cross_entropy_with_logits(
            input=mask_pred_logits, target=masks, weight=edge_masks, reduction='none'
        )
        edge_loss_val = edge_bce_map.mean() * self.edge_lambda

        # --- 6.3. Segmentation Penalty Logic (DDP Friendly) ---
        # Determine if the segmentation prediction implies a tampered image.
        min_tamper_signal_for_seg = max(1.0, 0.0001 * self.input_size * self.input_size)
        seg_predicts_tamper_binary = (mask_pred_prob.sum(dim=[1,2,3]) > min_tamper_signal_for_seg).float() # Shape: (N,)
        is_tampered_gt_squeezed = is_tampered_gt.squeeze(1) # Shape: (N,)

        # Apply penalty for False Positives (authentic image, but seg predicts tamper).
        fp_penalty_factor = torch.where(
            (is_tampered_gt_squeezed == 0) & (seg_predicts_tamper_binary == 1),
            self.seg_penalty_fp, torch.ones_like(seg_predicts_tamper_binary) 
        )
        # Apply penalty for False Negatives (tampered image, but seg predicts authentic).
        fn_penalty_factor = torch.where(
            (is_tampered_gt_squeezed == 1) & (seg_predicts_tamper_binary == 0),
            self.seg_penalty_fn, torch.ones_like(seg_predicts_tamper_binary) 
        )

        # Apply penalties and compute final mean segmentation losses.
        penalized_focal_loss_ps = focal_loss_ps * fp_penalty_factor * fn_penalty_factor
        penalized_dice_loss_ps = dice_loss_ps * fp_penalty_factor * fn_penalty_factor
        final_focal_loss = penalized_focal_loss_ps.mean()
        final_dice_loss = penalized_dice_loss_ps.mean()
        
        # --- 6.4. Consistency Loss ---
        # Enforces agreement between the classification head and the segmentation prediction's implication.
        consistency_loss_val = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.consistency_loss_weight > 0 and self.training:
            consistency_loss_val = self.consistency_loss_fn(
                tamper_logits.squeeze(1), # Classification head prediction (N,)
                seg_predicts_tamper_binary # Segmentation-implied prediction (N,) 
            ) * self.consistency_loss_weight
        
        # --- 6.5. Total Loss ---
        total_loss = (
            final_focal_loss +
            self.dice_weight * final_dice_loss +
            edge_loss_val +
            classification_loss +
            consistency_loss_val
        )

        return (total_loss, mask_pred_prob, tamper_pred_prob,
                final_focal_loss, edge_loss_val, final_dice_loss,
                classification_loss, consistency_loss_val)