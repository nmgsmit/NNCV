import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_NAME = "segformer-robust-v3"
MODEL_DESCRIPTION = (
    "Robust SegFormer v3 keeps the MSFE-FPN decoder with SyncBN and the strong v2 "
    "training and inference setup, but upgrades the augmentation policy with heavier "
    "weather, appearance, blur, compression, glare, vignette, and occlusion corruptions."
)
LAYER_NORM_EPS = 1e-6

MODEL_VARIANTS = {
    "b0": {
        "embed_dims": (32, 64, 160, 256),
        "depths": (2, 2, 2, 2),
        "sr_ratios": (8, 4, 2, 1),
        "num_heads": (1, 2, 5, 8),
        "mlp_ratios": (4.0, 4.0, 4.0, 4.0),
        "decoder_embedding_dim": 256,
        "drop_path_rate": 0.1,
    },
    "b5": {
        "embed_dims": (64, 128, 320, 512),
        "depths": (3, 6, 40, 3),
        "sr_ratios": (8, 4, 2, 1),
        "num_heads": (1, 2, 5, 8),
        "mlp_ratios": (4.0, 4.0, 4.0, 4.0),
        "decoder_embedding_dim": 768,
        "drop_path_rate": 0.1,
    },
}


def get_model_variant_config(variant: str) -> dict:
    try:
        return MODEL_VARIANTS[variant]
    except KeyError as exc:
        available = ", ".join(MODEL_VARIANTS)
        raise ValueError(f"Unknown SegFormer variant '{variant}'. Available variants: {available}.") from exc


def infer_model_variant_from_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    stage1_weight = state_dict.get("encoder.stage1.patch_embed.proj.weight")
    decoder_proj = state_dict.get("decode_head.lateral_c1.conv.weight")

    if torch.is_tensor(stage1_weight):
        stage1_dim = int(stage1_weight.shape[0])
        for variant, config in MODEL_VARIANTS.items():
            if config["embed_dims"][0] == stage1_dim:
                return variant

    if torch.is_tensor(decoder_proj):
        decoder_dim = int(decoder_proj.shape[1])
        for variant, config in MODEL_VARIANTS.items():
            if config["embed_dims"][0] == decoder_dim:
                return variant

    raise KeyError("Unable to infer the SegFormer variant from the checkpoint state dict.")


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int, stride: int):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim, eps=LAYER_NORM_EPS)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, height, width


class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        sr_ratio: int = 1,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim, eps=LAYER_NORM_EPS)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape
        q = self.q(x).reshape(batch_size, num_tokens, self.num_heads, channels // self.num_heads)
        q = q.permute(0, 2, 1, 3)

        if self.sr is not None:
            x_ = x.transpose(1, 2).reshape(batch_size, channels, height, width)
            x_ = self.sr(x_).reshape(batch_size, channels, -1).transpose(1, 2)
            x_ = self.norm(x_)
        else:
            x_ = x

        kv = self.kv(x_).reshape(batch_size, -1, 2, self.num_heads, channels // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        key, value = kv[0], kv[1]

        attention = (q @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention)

        x = (attention @ value).transpose(1, 2).reshape(batch_size, num_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MixFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_dim,
        )
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = self.fc1(x)
        batch_size, _, channels = x.shape
        x = x.transpose(1, 2).reshape(batch_size, channels, height, width)
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
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        sr_ratio: int = 1,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=LAYER_NORM_EPS)
        self.attn = EfficientSelfAttention(
            dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=LAYER_NORM_EPS)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio), drop=mlp_drop)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        x = x + self.drop_path(self.mlp(self.norm2(x), height, width))
        return x


class MixVisionStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        sr_ratio: int,
        mlp_ratio: float,
        patch_size: int,
        stride: int,
        drop_path_rates: list[float],
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(in_channels, embed_dim, patch_size, stride)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    sr_ratio=sr_ratio,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    mlp_drop=mlp_drop,
                    drop_path=drop_path_rates[index],
                )
                for index in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=LAYER_NORM_EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, height, width = self.patch_embed(x)
        for block in self.blocks:
            x = block(x, height, width)
        x = self.norm(x)
        batch_size, _, channels = x.shape
        return x.transpose(1, 2).reshape(batch_size, channels, height, width)


class MixVisionTransformerEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dims: tuple[int, int, int, int],
        depths: tuple[int, int, int, int],
        num_heads: tuple[int, int, int, int],
        sr_ratios: tuple[int, int, int, int],
        mlp_ratios: tuple[float, float, float, float],
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
        drop_path_rate: float = 0.1,
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
            in_channels=in_channels,
            embed_dim=embed_dims[0],
            depth=depths[0],
            num_heads=num_heads[0],
            sr_ratio=sr_ratios[0],
            mlp_ratio=mlp_ratios[0],
            qkv_bias=qkv_bias,
            patch_size=7,
            stride=4,
            drop_path_rates=stage_slices[0],
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_drop=mlp_drop,
        )
        self.stage2 = MixVisionStage(
            in_channels=embed_dims[0],
            embed_dim=embed_dims[1],
            depth=depths[1],
            num_heads=num_heads[1],
            sr_ratio=sr_ratios[1],
            mlp_ratio=mlp_ratios[1],
            qkv_bias=qkv_bias,
            patch_size=3,
            stride=2,
            drop_path_rates=stage_slices[1],
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_drop=mlp_drop,
        )
        self.stage3 = MixVisionStage(
            in_channels=embed_dims[1],
            embed_dim=embed_dims[2],
            depth=depths[2],
            num_heads=num_heads[2],
            sr_ratio=sr_ratios[2],
            mlp_ratio=mlp_ratios[2],
            qkv_bias=qkv_bias,
            patch_size=3,
            stride=2,
            drop_path_rates=stage_slices[2],
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_drop=mlp_drop,
        )
        self.stage4 = MixVisionStage(
            in_channels=embed_dims[2],
            embed_dim=embed_dims[3],
            depth=depths[3],
            num_heads=num_heads[3],
            sr_ratio=sr_ratios[3],
            mlp_ratio=mlp_ratios[3],
            qkv_bias=qkv_bias,
            patch_size=3,
            stride=2,
            drop_path_rates=stage_slices[3],
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_drop=mlp_drop,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        return c1, c2, c3, c4


class ConvSyncBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int | None = None,
        activation: bool = True,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.norm = nn.SyncBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class PyramidPoolingModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_scales: tuple[int, ...] = (2, 3, 6, 8),
    ):
        super().__init__()
        reduction_dim = max(out_channels // len(pool_scales), 32)
        self.reduce = ConvSyncBNAct(in_channels, out_channels, kernel_size=1, padding=0)
        self.pool_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    ConvSyncBNAct(out_channels, reduction_dim, kernel_size=1, padding=0),
                )
                for scale in pool_scales
            ]
        )
        fusion_channels = out_channels + len(pool_scales) * reduction_dim
        self.fuse = ConvSyncBNAct(fusion_channels, out_channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        height, width = x.shape[-2:]
        pooled_features = [x]
        for branch in self.pool_branches:
            pooled = branch(x)
            pooled = F.interpolate(pooled, size=(height, width), mode="bilinear", align_corners=False)
            pooled_features.append(pooled)
        return self.fuse(torch.cat(pooled_features, dim=1))


class ScaleAttentionFusion(nn.Module):
    def __init__(self, channels: int, num_levels: int):
        super().__init__()
        self.attention_reduce = ConvSyncBNAct(channels * num_levels, channels, kernel_size=1, padding=0)
        self.attention_pred = nn.Conv2d(channels, num_levels, kernel_size=1)
        self.refine = ConvSyncBNAct(channels, channels, kernel_size=3)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        if len(features) == 0:
            raise ValueError("ScaleAttentionFusion requires at least one feature map.")

        stacked = torch.stack(features, dim=1)
        attention_logits = self.attention_pred(self.attention_reduce(torch.cat(features, dim=1)))
        attention = torch.softmax(attention_logits, dim=1).unsqueeze(2)
        fused = (stacked * attention).sum(dim=1)
        return self.refine(fused)


class MSFEFPNHead(nn.Module):
    def __init__(
        self,
        in_channels: tuple[int, int, int, int],
        embedding_dim: int,
        n_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.lateral_c1 = ConvSyncBNAct(in_channels[0], embedding_dim, kernel_size=1, padding=0)
        self.lateral_c2 = ConvSyncBNAct(in_channels[1], embedding_dim, kernel_size=1, padding=0)
        self.lateral_c3 = ConvSyncBNAct(in_channels[2], embedding_dim, kernel_size=1, padding=0)
        self.context_c4 = PyramidPoolingModule(in_channels[3], embedding_dim)

        self.refine_c4 = ConvSyncBNAct(embedding_dim, embedding_dim, kernel_size=3)
        self.refine_c3 = ConvSyncBNAct(embedding_dim, embedding_dim, kernel_size=3)
        self.refine_c2 = ConvSyncBNAct(embedding_dim, embedding_dim, kernel_size=3)
        self.refine_c1 = ConvSyncBNAct(embedding_dim, embedding_dim, kernel_size=3)

        self.scale_attention = ScaleAttentionFusion(embedding_dim, num_levels=4)
        self.fuse = ConvSyncBNAct(embedding_dim * 5, embedding_dim, kernel_size=3)
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(embedding_dim, n_classes, kernel_size=1)

    def forward(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        c1, c2, c3, c4 = features
        target_size = c1.shape[-2:]

        p4 = self.refine_c4(self.context_c4(c4))
        p3 = self.refine_c3(
            self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        )
        p2 = self.refine_c2(
            self.lateral_c2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        )
        p1 = self.refine_c1(
            self.lateral_c1(c1) + F.interpolate(p2, size=target_size, mode="bilinear", align_corners=False)
        )

        aligned_features = [
            p1,
            F.interpolate(p2, size=target_size, mode="bilinear", align_corners=False),
            F.interpolate(p3, size=target_size, mode="bilinear", align_corners=False),
            F.interpolate(p4, size=target_size, mode="bilinear", align_corners=False),
        ]
        attended_feature = self.scale_attention(aligned_features)
        x = self.fuse(torch.cat(aligned_features + [attended_feature], dim=1))
        x = self.dropout(x)
        return self.classifier(x)


class Model(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 19,
        variant: str = "b5",
        dropout: float = 0.1,
        attn_drop: float = 0.0,
        mlp_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.variant = variant
        self.config = get_model_variant_config(variant)

        self.encoder = MixVisionTransformerEncoder(
            in_channels=in_channels,
            embed_dims=self.config["embed_dims"],
            depths=self.config["depths"],
            num_heads=self.config["num_heads"],
            sr_ratios=self.config["sr_ratios"],
            mlp_ratios=self.config["mlp_ratios"],
            qkv_bias=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_drop=mlp_drop,
            drop_path_rate=self.config["drop_path_rate"],
        )
        self.decode_head = MSFEFPNHead(
            in_channels=self.config["embed_dims"],
            embedding_dim=self.config["decoder_embedding_dim"],
            n_classes=n_classes,
            dropout=dropout,
        )
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def _extract_checkpoint_state(self, checkpoint: dict) -> dict:
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

        for nested_key in ("state_dict", "model", "module"):
            nested_state = checkpoint.get(nested_key)
            if isinstance(nested_state, dict):
                return nested_state
        return checkpoint

    def _remap_pretrained_key(self, key: str) -> str | None:
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

    def load_pretrained(self, folder_path: str) -> None:
        weight_path = os.path.join(folder_path, "pytorch_model.bin")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Missing pretrained weights at {weight_path}")

        checkpoint = torch.load(weight_path, map_location="cpu")
        checkpoint_state = self._extract_checkpoint_state(checkpoint)
        current_state = self.state_dict()
        remapped_state: dict[str, torch.Tensor] = {}
        kv_buffers: dict[str, dict[str, torch.Tensor]] = {}

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, but got {x.shape[1]}")

        input_size = x.shape[-2:]
        features = self.encoder(x)
        logits = self.decode_head(features)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits
