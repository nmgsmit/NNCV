
import math
from functools import partial

import torch
import torch.nn as nn

def drop_path(x, drop_prob: float = 0., training: bool = False):
    
    """
    Applies path dropout to the input tensor `x` during training.

    This function implements a stochastic regularization technique where
    elements of the input tensor `x` are randomly set to zero with a probability
    `drop_prob`, helping to prevent overfitting. This is particularly useful
    for regularizing large neural networks. The dropout is only applied if
    `training` is True. Otherwise, the input tensor is returned unchanged.
    The function supports tensors of any dimensionality.

    Parameters:
    - x (Tensor): The input tensor to which dropout will be applied.
    - drop_prob (float, optional): The probability of dropping a path. Defaults to 0.
    - training (bool, optional): Flag indicating whether dropout should be applied,
      i.e., if the model is in training mode. Defaults to False.

    Returns:
    - Tensor: The output tensor after applying dropout, if training is True and
      drop_prob > 0. Otherwise, the original input tensor is returned.

    Note:
    - The function adjusts the values of the retained elements to maintain the
      expected sum of the input, compensating for the dropped paths.
    """
     
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """
    Implements a simple Multilayer Perceptron (MLP) module.

    Parameters:
    - in_features (int): Number of features in the input tensor.
    - hidden_features (int, optional): Number of features in the hidden layer. Defaults to `in_features` if not specified.
    - out_features (int, optional): Number of features in the output tensor. Defaults to `in_features` if not specified.
    - act_layer (nn.Module, optional): Activation layer to use. Defaults to `nn.GELU`.
    - drop (float, optional): Dropout probability for the dropout layer. Defaults to 0.

    The module consists of two fully connected layers with an activation layer in between, followed by dropout applied after each of the fully connected layers.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    """
    Implements a multi-head self-attention mechanism with optional scaling.

    This module computes self-attention, using a scaled dot-product mechanism, over input features. It supports optional biases in the query, key, and value projections, scaling of the attention scores, and dropout in both the attention scores and the output projection.

    Parameters:
    - dim (int): Dimensionality of the input features and the output features.
    - num_heads (int, optional): Number of attention heads. Defaults to 8.
    - qkv_bias (bool, optional): If True, adds a learnable bias to query, key, and value projections. Defaults to False.
    - qk_scale (float, optional): Scale factor for query-key dot products. If None, defaults to dim ** -0.5. When specified, overrides the default scaling.
    - attn_drop (float, optional): Dropout rate for attention weights. Defaults to 0.
    - proj_drop (float, optional): Dropout rate for the output of the final projection layer. Defaults to 0.

    The forward pass accepts an input tensor `x` and returns the transformed tensor and the attention weights. The input tensor is expected to have the shape (batch_size, num_features, dim), where `num_features` is the number of features (or tokens) and `dim` is the feature dimensionality.

    The output consists of the transformed input tensor with the same shape as the input and the attention weights tensor of shape (batch_size, num_heads, num_features, num_features), representing the attention scores applied to the input features.
    """
    

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads # Number of heads in multi-headed attention
        self.scale = qk_scale or head_dim ** -0.5 #sqrt(Dk)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # single linear layer that projects input into QKV
        self.attn_drop = nn.Dropout(attn_drop) # dropout rate for attentin weights
        self.proj = nn.Linear(dim, dim) # final projection layer
        self.proj_drop = nn.Dropout(proj_drop) #dropout rate for output of final projection layer

    def forward(self, x):
        B, N, C = x.shape # extract from input: B= atch size, N tokens/patches, C= Channel/Embedding dimension
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) #get qkv matrix dependent on num heads, batch size, patches, embedding dimension 

        q, k, v = qkv
        kT = k.transpose(-2,-1)
        attn = ((q @ kT)*self.scale) #MatMul, Scale

        attn = attn.softmax(dim=-1) # Softmax
        attn = self.attn_drop(attn) #regularization dropout

        x = (attn @ v) #matmul
        x = x.transpose(1,2) # Put heads and sequence in normal position
        x = x.reshape(B, N, C) # concatenate information of all attention heads

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):

    """
    A Transformer block module combining self-attention and a feed-forward network.

    This module applies the following operations in sequence:
    1. Normalization of the input.
    2. Multi-head self-attention mechanism.
    3. Addition of the attention output to the input (residual connection) and optional dropout.
    4. Normalization of the residual output.
    5. A feed-forward network applied to the normalized output.
    6. Addition of the feed-forward network output to the normalized output (residual connection) and optional dropout.

    Parameters:
    - dim (int): Dimensionality of the input features.
    - num_heads (int): Number of attention heads.
    - mlp_ratio (float, optional): Ratio of the hidden dimension size of the MLP module to the input dimension size. Defaults to 4.
    - qkv_bias (bool, optional): If set to True, adds a learnable bias to query, key, and value. Defaults to False.
    - qk_scale (float, optional): Override scale factor for qk. Defaults to None, which uses dim ** -0.5.
    - drop (float, optional): Dropout rate for attention probabilities and the output of the MLP. Defaults to 0.
    - attn_drop (float, optional): Dropout rate for attention weights. Defaults to 0.
    - drop_path (float, optional): Dropout rate for paths in the model. Defaults to 0.
    - act_layer (torch.nn.Module, optional): Activation function used in the MLP module. Defaults to nn.GELU.
    - norm_layer (torch.nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.

    The module supports returning the attention matrix from the self-attention mechanism if `return_attention` is set to True during the forward pass.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """
    Converts an image into a sequence of patches and embeds them.

    This module uses a convolutional layer to transform the input images into a flat sequence of embeddings, 
    effectively converting each patch of the image into an embedding vector.

    Parameters:
    - img_size (int, optional): Size of the input image (height and width). Defaults to 224.
    - patch_size (int, optional): Size of each patch (height and width). Defaults to 16.
    - in_chans (int, optional): Number of input channels (e.g., 3 for RGB images). Defaults to 3.
    - embed_dim (int, optional): Dimension of the patch embeddings. Defaults to 768.

    The module calculates the number of patches by dividing the image size by the patch size, both vertically and horizontally. 
    It then applies a 2D convolutional layer to project each patch to the embedding space defined by `embed_dim`.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        
        x = self.proj(x).flatten(2).transpose(1, 2)

        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
       
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

