# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size = 224, patch_size=16, en_embed_dim=768, en_depth=12, en_num_heads=12,
        dc_embed_dim=512, dc_depth=8, dc_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), in_chans = 3, norm_pix_loss=False):
        super().__init__()
        # ===================================================================
        embed_dim = en_embed_dim
        depth = en_depth
        num_heads = en_num_heads
        # ------------------------------ PatchEmbed ---------------
        self.en_patch_embed = PatchEmbed(img_size, patch_size, in_chans = in_chans, embed_dim = embed_dim) 
        num_patches = self.en_patch_embed.num_patches
        # ------------------------------ class token ---------------
        self.en_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.normal_(self.en_cls_token, std=.02)
        # ------------------------------ Positional embed ----------
        self.en_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # ------------------------------ Encoder --- ---------------
        self.en_blocks = nn.ModuleList([
            Block(dim = embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        # ------------------------------ normal layer --------------
        self.en_norm = norm_layer(embed_dim)
        # ------------------------------ head --------------
        # ===================================================================
        self.dc_embed = nn.Linear(in_features = en_embed_dim, out_features = dc_embed_dim, bias=True)
        # ===================================================================
        embed_dim = dc_embed_dim
        depth = dc_depth
        num_heads = dc_num_heads
        self.num_classes = patch_size**2 * in_chans
        depth = dc_depth
        # ------------------------------ class token ---------------
        self.dc_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.normal_(self.dc_cls_token, std=.02)
        # ------------------------------ Positional embed ----------
        self.dc_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # ------------------------------ Encoder --- ---------------
        self.dc_blocks = nn.ModuleList([
            Block(dim = embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        # ------------------------------ normal layer --------------
        self.dc_norm = norm_layer(embed_dim)
        # ------------------------------ head --------------
        self.dc_head = nn.Linear(embed_dim, self.num_classes, bias=True) if self.num_classes > 0 else nn.Identity()
        
        self.init()
        
    def init(self):
        en_pos_embed = get_2d_sincos_pos_embed(self.en_pos_embed.shape[-1], int(self.en_patch_embed.num_patches**.5), cls_token=True)
        self.en_pos_embed.data.copy_(torch.from_numpy(en_pos_embed).float().unsqueeze(0))
        dc_pos_embed = get_2d_sincos_pos_embed(self.dc_pos_embed.shape[-1], int(self.en_patch_embed.num_patches**.5), cls_token=True)
        self.dc_pos_embed.data.copy_(torch.from_numpy(dc_pos_embed).float().unsqueeze(0))     
        ####### do we need to initialise conv_2d in PatchEmbed
        self.apply(self._init_weights)
        
        #self.en_pos_embed = build_2d_sincos_position_embedding(*self.en_pos_embed.shape, self.en_pos_embed.shape[-1])
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def patchify(self, imgs):
        pass

    def unpatchify(self, x):
        pass
    def random_masking(self, x, mask_ratio):
        pass

    def en_forward(self, x):
        x = self.en_patch_embed(x)
        x = x + self.en_pos_embed[:, 1:, :]
        #--------- masking --
        #--------------------
        #cls_token = self.en_cls_token + self.en_pos_embed[:, 0, :]
        #cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        #x = torch.cat((cls_tokens, x), dim=1)
        #------- alternatinve ---
        cls_tokens = self.en_cls_token.expand(x.shape[0], -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.en_pos_embed  
        for block in self.en_blocks:
            x = block(x)
        x = self.en_norm(x)
        return x
    
    def dc_forward(self, x):
        x = self.dc_embed(x)
        #---------- masking 
        #---------
        x = x + self.dc_pos_embed 
        for block in self.dc_blocks:
            x = block(x)
        x = self.dc_norm(x)
        x = self.dc_head(x)
        return x[:, 1:, :]

    def forward_loss(self, imgs, pred, mask):
        b, c, H, W = imgs.shape
        p = int(self.en_patch_embed.patch_size[0])
        x = imgs.view(b, c, H // p, p, W // p, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(b, H*W // p**2, p**2 * 3))
        y = pred
        loss = F.mse_loss(x, y) # no mask is here
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        x = self.en_forward(imgs)
        pred = self.dc_forward(x)
        mask = None
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, en_embed_dim=768, en_depth=12, en_num_heads=12,
        dc_embed_dim=512, dc_depth=8, dc_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),  **kwargs)

    return model



# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
