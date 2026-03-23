import os

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, stride):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w


class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=1, sr_ratio=1, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

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
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

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
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, h, w):
        x = self.fc1(x)
        b, _, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        sr_ratio=1,
        attn_drop=0.0,
        proj_drop=0.0,
        mlp_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(
            dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio), drop=mlp_drop)

    def forward(self, x, h, w):
        x = x + self.drop_path(self.attn(self.norm1(x), h, w))
        x = x + self.drop_path(self.mlp(self.norm2(x), h, w))
        return x


class MixVisionStage(nn.Module):
    def __init__(
        self,
        in_channels,
        embed_dim,
        depth,
        num_heads,
        sr_ratio,
        patch_size,
        stride,
        drop_path_rates,
        attn_drop=0.0,
        proj_drop=0.0,
        mlp_drop=0.0,
    ):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(in_channels, embed_dim, patch_size, stride)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    num_heads=num_heads,
                    sr_ratio=sr_ratio,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    mlp_drop=mlp_drop,
                    drop_path=drop_path_rates[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x, h, w = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x, h, w)
        x = self.norm(x)
        b, _, c = x.shape
        return x.transpose(1, 2).reshape(b, c, h, w)


class MixVisionTransformerEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        embed_dims,
        depths,
        num_heads,
        sr_ratios,
        attn_drop=0.0,
        proj_drop=0.0,
        mlp_drop=0.0,
        drop_path_rate=0.1,
    ):
        super().__init__()

        total_depth = sum(depths)
        drop_path_rates = torch.linspace(0, drop_path_rate, total_depth).tolist()

        stage_slices = []
        start = 0
        for depth in depths:
            stage_slices.append(drop_path_rates[start:start + depth])
            start += depth

        self.stage1 = MixVisionStage(
            in_channels,
            embed_dims[0],
            depths[0],
            num_heads[0],
            sr_ratios[0],
            patch_size=7,
            stride=4,
            drop_path_rates=stage_slices[0],
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_drop=mlp_drop,
        )
        self.stage2 = MixVisionStage(
            embed_dims[0],
            embed_dims[1],
            depths[1],
            num_heads[1],
            sr_ratios[1],
            patch_size=3,
            stride=2,
            drop_path_rates=stage_slices[1],
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_drop=mlp_drop,
        )
        self.stage3 = MixVisionStage(
            embed_dims[1],
            embed_dims[2],
            depths[2],
            num_heads[2],
            sr_ratios[2],
            patch_size=3,
            stride=2,
            drop_path_rates=stage_slices[2],
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_drop=mlp_drop,
        )
        self.stage4 = MixVisionStage(
            embed_dims[2],
            embed_dims[3],
            depths[3],
            num_heads[3],
            sr_ratios[3],
            patch_size=3,
            stride=2,
            drop_path_rates=stage_slices[3],
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_drop=mlp_drop,
        )

    def forward(self, x):
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        return c1, c2, c3, c4


class SegFormerHead(nn.Module):
    def __init__(self, in_channels, embedding_dim, n_classes, dropout=0.1):
        super().__init__()

        self.linear_c1 = nn.Conv2d(in_channels[0], embedding_dim, kernel_size=1)
        self.linear_c2 = nn.Conv2d(in_channels[1], embedding_dim, kernel_size=1)
        self.linear_c3 = nn.Conv2d(in_channels[2], embedding_dim, kernel_size=1)
        self.linear_c4 = nn.Conv2d(in_channels[3], embedding_dim, kernel_size=1)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(embedding_dim, n_classes, kernel_size=1)

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
        return self.classifier(x)


class Model(nn.Module):
    def __init__(
        self,
        in_channels=3,
        n_classes=19,
        embed_dims=(64, 128, 320, 512),
        depths=(3, 6, 40, 3),
        sr_ratios=(8, 4, 2, 1),
        num_heads=(1, 2, 5, 8),
        decoder_embedding_dim=768,
        dropout=0.1,
        attn_drop=0.0,
        mlp_drop=0.0,
        proj_drop=0.0,
        drop_path_rate=0.1,
    ):
        super().__init__()

        self.encoder = MixVisionTransformerEncoder(
            in_channels=in_channels,
            embed_dims=embed_dims,
            depths=depths,
            num_heads=num_heads,
            sr_ratios=sr_ratios,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_drop=mlp_drop,
            drop_path_rate=drop_path_rate,
        )

        self.decode_head = SegFormerHead(
            in_channels=embed_dims,
            embedding_dim=decoder_embedding_dim,
            n_classes=n_classes,
            dropout=dropout,
        )

    def _extract_checkpoint_state(self, checkpoint):
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

        for nested_key in ("state_dict", "model", "module"):
            nested_state = checkpoint.get(nested_key)
            if isinstance(nested_state, dict):
                return nested_state
        return checkpoint

    def _remap_pretrained_key(self, key):
        if key.startswith("module."):
            key = key[len("module."):]

        valid_prefixes = (
            "segformer.encoder.",
            "encoder.",
            "backbone.",
            "segformer.backbone.",
        )

        source_prefix = next((prefix for prefix in valid_prefixes if key.startswith(prefix)), None)
        if source_prefix is None:
            return None

        key = f"encoder.{key[len(source_prefix):]}"

        for stage_idx in range(4):
            key = key.replace(f"patch_embeddings.{stage_idx}.proj", f"stage{stage_idx + 1}.patch_embed.proj")
            key = key.replace(
                f"patch_embeddings.{stage_idx}.layer_norm",
                f"stage{stage_idx + 1}.patch_embed.norm",
            )
            key = key.replace(f"layer_norm.{stage_idx}", f"stage{stage_idx + 1}.norm")
            key = key.replace(f"block.{stage_idx}", f"stage{stage_idx + 1}.blocks")

        key = key.replace(".layer_norm_1", ".norm1")
        key = key.replace(".layer_norm_2", ".norm2")
        key = key.replace(".attention.self.query", ".attn.q")
        key = key.replace(".attention.self.sr", ".attn.sr")
        key = key.replace(".attention.self.layer_norm", ".attn.norm")
        key = key.replace(".attention.output.dense", ".attn.proj")
        key = key.replace(".mlp.dense1", ".mlp.fc1")
        key = key.replace(".mlp.dwconv.dwconv", ".mlp.dwconv")
        key = key.replace(".mlp.dense2", ".mlp.fc2")
        return key

    def load_pretrained(self, folder_path):
        weight_path = os.path.join(folder_path, "pytorch_model.bin")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Missing weights at {weight_path}")

        checkpoint = torch.load(weight_path, map_location="cpu")
        checkpoint_state = self._extract_checkpoint_state(checkpoint)
        current_state = self.state_dict()
        remapped_state = {}
        kv_buffers = {}

        for source_key, value in checkpoint_state.items():
            target_key = self._remap_pretrained_key(source_key)
            if target_key is None:
                continue

            if source_key.endswith(".attention.self.key.weight"):
                fused_key = target_key.replace(".attn.q.weight", ".attn.kv.weight").replace(
                    ".attention.self.key.weight",
                    ".attn.kv.weight",
                )
                kv_buffers.setdefault(fused_key, {})["key"] = value
                continue
            if source_key.endswith(".attention.self.value.weight"):
                fused_key = target_key.replace(".attention.self.value.weight", ".attn.kv.weight")
                kv_buffers.setdefault(fused_key, {})["value"] = value
                continue
            if source_key.endswith(".attention.self.key.bias"):
                fused_key = target_key.replace(".attn.q.bias", ".attn.kv.bias").replace(
                    ".attention.self.key.bias",
                    ".attn.kv.bias",
                )
                kv_buffers.setdefault(fused_key, {})["key"] = value
                continue
            if source_key.endswith(".attention.self.value.bias"):
                fused_key = target_key.replace(".attention.self.value.bias", ".attn.kv.bias")
                kv_buffers.setdefault(fused_key, {})["value"] = value
                continue

            if target_key in current_state and current_state[target_key].shape == value.shape:
                remapped_state[target_key] = value

        for fused_key, pieces in kv_buffers.items():
            if "key" not in pieces or "value" not in pieces:
                continue
            fused_value = torch.cat([pieces["key"], pieces["value"]], dim=0)
            if fused_key in current_state and current_state[fused_key].shape == fused_value.shape:
                remapped_state[fused_key] = fused_value

        missing_encoder_keys = sorted(
            key for key in current_state.keys()
            if key.startswith("encoder.") and key not in remapped_state
        )

        msg = self.load_state_dict(remapped_state, strict=False)
        print(f"Loaded {len(remapped_state)} pretrained tensors from {weight_path}")
        print(f"Pretrained encoder load result: {msg}")
        if missing_encoder_keys:
            preview_count = min(10, len(missing_encoder_keys))
            print(
                f"Encoder keys still randomly initialized: {len(missing_encoder_keys)} "
                f"(showing first {preview_count})"
            )
            for key in missing_encoder_keys[:preview_count]:
                print(f"  - {key}")
            if len(missing_encoder_keys) > preview_count:
                print("  - ...")

    def forward(self, x):
        input_size = x.shape[-2:]
        features = self.encoder(x)

        main_logits = self.decode_head(features)
        main_logits = F.interpolate(main_logits, size=input_size, mode="bilinear", align_corners=False)
        return main_logits
