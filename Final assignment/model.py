import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# ---------------------------
# DropPath (stochastic depth)
# ---------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ---------------------------
# Encoder Components
# ---------------------------
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, stride):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w

class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=1, sr_ratio=1, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None

    def forward(self, x, h, w):
        b, n, c = x.shape

        q = self.q(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)

        if self.sr is not None:
            x_ = x.transpose(1, 2).reshape(b, c, h, w)
            x_ = self.sr(x_).reshape(b, c, -1).transpose(1, 2)
            x_ = self.norm(x_)
        else:
            x_ = x

        kv = self.kv(x_).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class MixFFN(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, h, w):
        x = self.fc1(x)
        b, n, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, sr_ratio=1, drop=0.0, drop_path=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio, drop, drop)

        self.drop_path = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio), drop)

    def forward(self, x, h, w):
        x = x + self.drop_path(self.attn(self.norm1(x), h, w))
        x = x + self.drop_path(self.mlp(self.norm2(x), h, w))
        return x

class MixVisionStage(nn.Module):
    def __init__(self, in_channels, embed_dim, depth, num_heads, sr_ratio, patch_size, stride, drop_rate):
        super().__init__()

        self.patch_embed = OverlapPatchEmbed(in_channels, embed_dim, patch_size, stride)

        dpr = torch.linspace(0, 0.1, depth).tolist()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, sr_ratio=sr_ratio, drop=drop_rate, drop_path=dpr[i])
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x, h, w = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x, h, w)
        x = self.norm(x)
        b, n, c = x.shape
        return x.transpose(1, 2).reshape(b, c, h, w)

class MixVisionTransformerEncoder(nn.Module):
    def __init__(self, in_channels, embed_dims, depths, num_heads, sr_ratios, drop_rate=0.1):
        super().__init__()

        self.stage1 = MixVisionStage(in_channels, embed_dims[0], depths[0], num_heads[0], sr_ratios[0], 7, 4, drop_rate)
        self.stage2 = MixVisionStage(embed_dims[0], embed_dims[1], depths[1], num_heads[1], sr_ratios[1], 3, 2, drop_rate)
        self.stage3 = MixVisionStage(embed_dims[1], embed_dims[2], depths[2], num_heads[2], sr_ratios[2], 3, 2, drop_rate)
        self.stage4 = MixVisionStage(embed_dims[2], embed_dims[3], depths[3], num_heads[3], sr_ratios[3], 3, 2, drop_rate)

    def forward(self, x):
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        return c1, c2, c3, c4


# ---------------------------
# Decoder
# ---------------------------
class SegFormerHead(nn.Module):
    def __init__(self, in_channels, embedding_dim, n_classes, dropout=0.1):
        super().__init__()

        self.linear_c1 = nn.Conv2d(in_channels[0], embedding_dim, 1)
        self.linear_c2 = nn.Conv2d(in_channels[1], embedding_dim, 1)
        self.linear_c3 = nn.Conv2d(in_channels[2], embedding_dim, 1)
        self.linear_c4 = nn.Conv2d(in_channels[3], embedding_dim, 1)

        self.linear_fuse = nn.Conv2d(embedding_dim * 4, embedding_dim, 1)
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(embedding_dim, n_classes, 1)

    def forward(self, features):
        c1, c2, c3, c4 = features
        h, w = c1.shape[-2:]

        c1 = self.linear_c1(c1)
        c2 = F.interpolate(self.linear_c2(c2), size=(h, w), mode="bilinear", align_corners=False)
        c3 = F.interpolate(self.linear_c3(c3), size=(h, w), mode="bilinear", align_corners=False)
        c4 = F.interpolate(self.linear_c4(c4), size=(h, w), mode="bilinear", align_corners=False)

        x = torch.cat([c1, c2, c3, c4], dim=1)
        x = self.linear_fuse(x)
        x = self.dropout(x)
        x = self.classifier(x)

        return x


# ---------------------------
# Main Model Wrapper
# ---------------------------
class Model(nn.Module):
    def __init__(self, in_channels=3, n_classes=19,
                 embed_dims=(64, 128, 320, 512),
                 depths=(3, 6, 40, 3),
                 sr_ratios=(8, 4, 2, 1),
                 num_heads=(1, 2, 5, 8),
                 decoder_embedding_dim=768,
                 dropout=0.1):
        super().__init__()

        self.encoder = MixVisionTransformerEncoder(
            in_channels, embed_dims, depths, num_heads, sr_ratios, drop_rate=dropout
        )

        self.decode_head = SegFormerHead(
            in_channels=embed_dims,
            embedding_dim=decoder_embedding_dim,
            n_classes=n_classes,
            dropout=dropout,
        )

        self.aux_head = nn.Sequential(
            nn.Conv2d(embed_dims[2], embed_dims[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims[2], n_classes, 1)
        )

    def load_pretrained(self, folder_path):
        """Loads weights from HuggingFace local folder."""
        weight_path = os.path.join(folder_path, "pytorch_model.bin")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Missing weights at {weight_path}")

        state_dict = torch.load(weight_path, map_location="cpu")
        new_state_dict = {}
        
        for key, value in state_dict.items():
            if not key.startswith("segformer.encoder"):
                continue
            
            k = key.replace("segformer.encoder.", "encoder.")
            
            # Map Stages/Patch Embeddings
            k = k.replace("patch_embeddings.0", "stage1.patch_embed")
            k = k.replace("patch_embeddings.1", "stage2.patch_embed")
            k = k.replace("patch_embeddings.2", "stage3.patch_embed")
            k = k.replace("patch_embeddings.3", "stage4.patch_embed")
            
            # Map Layer Norms
            k = k.replace("layer_norm.0", "stage1.norm")
            k = k.replace("layer_norm.1", "stage2.norm")
            k = k.replace("layer_norm.2", "stage3.norm")
            k = k.replace("layer_norm.3", "stage4.norm")
            
            # Map Blocks
            k = k.replace("block.0", "stage1.blocks")
            k = k.replace("block.1", "stage2.blocks")
            k = k.replace("block.2", "stage3.blocks")
            k = k.replace("block.3", "stage4.blocks")
            
            # Map Attention sub-keys 
            k = k.replace(".attention.self.query", ".attn.q")
            k = k.replace(".attention.self.key", ".attn.kv") 
            k = k.replace(".attention.self.value", ".attn.kv") 
            k = k.replace(".attention.output.dense", ".attn.proj")
            
            # Map FFN/MLP sub-keys
            k = k.replace(".mlp.dense1", ".mlp.fc1")
            k = k.replace(".mlp.dwconv.dwconv", ".mlp.dwconv")
            k = k.replace(".mlp.dense2", ".mlp.fc2")

            if k in self.state_dict():
                new_state_dict[k] = value

        msg = self.load_state_dict(new_state_dict, strict=False)
        print(f"Pretrained encoder loaded: {msg}")

    def forward(self, x):
        input_size = x.shape[-2:]
        features = self.encoder(x)

        main_logits = self.decode_head(features)
        aux_logits = self.aux_head(features[2])

        main_logits = F.interpolate(main_logits, size=input_size, mode="bilinear", align_corners=False)
        aux_logits = F.interpolate(aux_logits, size=input_size, mode="bilinear", align_corners=False)

        if self.training:
            return main_logits, aux_logits
        return main_logits