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
import matplotlib.pyplot as plt
import numpy as np
class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, input_size = 224, patch_size=16, embed_dim=768, depth=1, num_heads=12,
        dc_embed_dim=512, dc_depth=1, dc_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), in_chans = 3, norm_pix_loss=False, mask_ratio= 0.75):
        super().__init__()
        # ===================================================================
        embed_dim = embed_dim
        depth = depth
        num_heads = num_heads
        # ------------------------------ PatchEmbed ---------------
        self.patch_embed = PatchEmbed(input_size, patch_size, in_chans = in_chans, embed_dim = embed_dim) 
        num_patches = self.patch_embed.num_patches
        # ------------------------------ class token ---------------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.normal_(self.cls_token, std=.02)
        # ------------------------------ Positional embed ----------
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # ------------------------------ Encoder --- ---------------
        self.blocks = nn.ModuleList([
            Block(dim = embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        # ------------------------------ normal layer --------------
        self.norm = norm_layer(embed_dim)
        # ------------------------------ head --------------
        #========================= masking =====================================
        self.Nmask = int(mask_ratio * num_patches)# Number of mmasked patches
        self.Nvis = num_patches - self.Nmask # Number of visible patches
        # ===================================================================
        self.dc_embed = nn.Linear(in_features = embed_dim, out_features = dc_embed_dim, bias=True)
        # ===================================================================
        embed_dim = dc_embed_dim
        depth = dc_depth
        num_heads = dc_num_heads
        self.num_classes = patch_size**2 * in_chans
        depth = dc_depth
        # ------------------------------ class token ---------------
        self.dc_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.normal_(self.dc_mask_token, std=.02)
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
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        dc_pos_embed = get_2d_sincos_pos_embed(self.dc_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.dc_pos_embed.data.copy_(torch.from_numpy(dc_pos_embed).float().unsqueeze(0))     
        ####### do we need to initialise conv_2d in PatchEmbed
        self.apply(self._init_weights)
        
        #self.pos_embed = build_2d_sincos_position_embedding(*self.pos_embed.shape, self.pos_embed.shape[-1])
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
            
    def show(self, img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')      

    def patchify(self, imgs):
        pass

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        pass

    def en_forward(self, x):
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)        
        return x
    
    def dc_forward(self, x):
        #---------
        x = x + self.dc_pos_embed 
        for block in self.dc_blocks:
            x = block(x)
        x = self.dc_norm(x)
        x = self.dc_head(x)
        return x[:, 1:, :]

    def forward_loss(self, imgs, pred, mask):
        b, c, H, W = imgs.shape
        p = int(self.patch_embed.patch_size[0])
        x = imgs.view(b, c, H // p, p, W // p, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(b, H*W // p**2, p**2 * 3))
        y = pred # B, 196 * 512
        
        mask = mask.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        x = x.gather(1, mask)
        y = y.gather(1, mask)
        
        loss = F.mse_loss(x, y) 

        return loss

    def forward(self, imgs, mask_ratio=0.75):
        b = imgs.shape[0]
        p = self.patch_embed.num_patches
        index = torch.rand(b, p).argsort(1).to(imgs.device)
        visInd = index[:, :self.Nvis]
        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]
        xVis= x.gather(1, visInd.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        #------------- encoder --------------
        xVis = self.en_forward(xVis)
        #--------- en to dc -----------------
        xVis = self.dc_embed(xVis)
        #------------- prepare --------------
        dc_cls_token = xVis[:,:1,:]
        
        xMasked = self.dc_mask_token.expand(b, self.Nmask, -1)
        xVis = xVis[:,1:,:]
        x_ = torch.cat((xVis, xMasked), dim=1)
        x_ = x_.gather(1, index.argsort(1).unsqueeze(-1).expand(-1, -1, xVis.shape[-1]))
        x = torch.cat([dc_cls_token, x_], dim=1)
        #------------- decoder --------------
        pred = self.dc_forward(x)
        loss = self.forward_loss(imgs, pred, mask=visInd)
        return loss, pred, visInd

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(input_size=224,
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        dc_embed_dim=512, dc_depth=1, dc_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),  **kwargs)

    return model



# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
