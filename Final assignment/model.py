import math

import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_NAME = "segformer-step4-mix-ffn"
MODEL_DESCRIPTION = (
    "Step 4: the single-stage transformer now combines overlap patch embedding, efficient "
    "self-attention, and Mix-FFN, while still using a simple decoder."
)
LAYER_NORM_EPS = 1e-6

MODEL_CONFIG = {
    "encoder_type": "single_stage",
    "decoder_type": "simple",
    "single_stage": {
        "patch_embed_type": "overlap",
        "attention_type": "efficient",
        "ffn_type": "mix",
        "embed_dim": 256,
        "depth": 6,
        "num_heads": 8,
        "patch_size": 21,
        "stride": 16,
        "mlp_ratio": 4.0,
        "sr_ratio": 4,
        "drop_path_rate": 0.1,
        "attn_drop": 0.0,
        "proj_drop": 0.0,
        "ffn_drop": 0.0,
    },
    "hierarchical": {
        "embed_dims": (32, 64, 160, 256),
        "depths": (2, 2, 2, 2),
        "num_heads": (1, 2, 5, 8),
        "sr_ratios": (8, 4, 2, 1),
        "mlp_ratios": (4.0, 4.0, 4.0, 4.0),
        "decoder_embedding_dim": 256,
        "drop_path_rate": 0.1,
        "attn_drop": 0.0,
        "proj_drop": 0.0,
        "ffn_drop": 0.0,
    },
}


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


class PlainPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int, stride: int):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=0,
        )
        self.norm = nn.LayerNorm(embed_dim, eps=LAYER_NORM_EPS)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, height, width


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


class StandardSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
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

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        del height, width

        batch_size, num_tokens, channels = x.shape
        q = self.q(x).reshape(batch_size, num_tokens, self.num_heads, channels // self.num_heads)
        q = q.permute(0, 2, 1, 3)

        kv = self.kv(x).reshape(batch_size, num_tokens, 2, self.num_heads, channels // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        key, value = kv[0], kv[1]

        attention = (q @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention)

        x = (attention @ value).transpose(1, 2).reshape(batch_size, num_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


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


class PlainFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        del height, width

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
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
        attention_type: str = "standard",
        ffn_type: str = "plain",
        sr_ratio: int = 1,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=LAYER_NORM_EPS)

        if attention_type == "standard":
            self.attn = StandardSelfAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
        elif attention_type == "efficient":
            self.attn = EfficientSelfAttention(
                dim,
                num_heads=num_heads,
                sr_ratio=sr_ratio,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")

        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=LAYER_NORM_EPS)

        if ffn_type == "plain":
            self.ffn = PlainFFN(dim, int(dim * mlp_ratio), drop=ffn_drop)
        elif ffn_type == "mix":
            self.ffn = MixFFN(dim, int(dim * mlp_ratio), drop=ffn_drop)
        else:
            raise ValueError(f"Unsupported FFN type: {ffn_type}")

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        x = x + self.drop_path(self.ffn(self.norm2(x), height, width))
        return x


class SingleStageTransformerEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        patch_size: int,
        stride: int,
        attention_type: str,
        ffn_type: str,
        patch_embed_type: str,
        mlp_ratio: float,
        sr_ratio: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        patch_cls = PlainPatchEmbed if patch_embed_type == "plain" else OverlapPatchEmbed
        self.patch_embed = patch_cls(in_channels, embed_dim, patch_size, stride)

        drop_path_rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attention_type=attention_type,
                    ffn_type=ffn_type,
                    sr_ratio=sr_ratio,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    ffn_drop=ffn_drop,
                    drop_path=drop_path_rates[index],
                )
                for index in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=LAYER_NORM_EPS)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        x, height, width = self.patch_embed(x)
        for block in self.blocks:
            x = block(x, height, width)
        x = self.norm(x)
        batch_size, _, channels = x.shape
        x = x.transpose(1, 2).reshape(batch_size, channels, height, width)
        return (x,)


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
        ffn_drop: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(in_channels, embed_dim, patch_size, stride)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attention_type="efficient",
                    ffn_type="mix",
                    sr_ratio=sr_ratio,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    ffn_drop=ffn_drop,
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
        drop_path_rate: float = 0.1,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffn_drop: float = 0.0,
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
            patch_size=7,
            stride=4,
            drop_path_rates=stage_slices[0],
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            ffn_drop=ffn_drop,
        )
        self.stage2 = MixVisionStage(
            in_channels=embed_dims[0],
            embed_dim=embed_dims[1],
            depth=depths[1],
            num_heads=num_heads[1],
            sr_ratio=sr_ratios[1],
            mlp_ratio=mlp_ratios[1],
            patch_size=3,
            stride=2,
            drop_path_rates=stage_slices[1],
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            ffn_drop=ffn_drop,
        )
        self.stage3 = MixVisionStage(
            in_channels=embed_dims[1],
            embed_dim=embed_dims[2],
            depth=depths[2],
            num_heads=num_heads[2],
            sr_ratio=sr_ratios[2],
            mlp_ratio=mlp_ratios[2],
            patch_size=3,
            stride=2,
            drop_path_rates=stage_slices[2],
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            ffn_drop=ffn_drop,
        )
        self.stage4 = MixVisionStage(
            in_channels=embed_dims[2],
            embed_dim=embed_dims[3],
            depth=depths[3],
            num_heads=num_heads[3],
            sr_ratio=sr_ratios[3],
            mlp_ratio=mlp_ratios[3],
            patch_size=3,
            stride=2,
            drop_path_rates=stage_slices[3],
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            ffn_drop=ffn_drop,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        return c1, c2, c3, c4


class SimpleDecoderHead(nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()
        self.classifier = nn.Conv2d(in_channels, n_classes, kernel_size=1)

    def forward(self, features: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.classifier(features[-1])


class SegFormerMLP(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        return self.proj(x)


class SegFormerHead(nn.Module):
    def __init__(
        self,
        in_channels: tuple[int, int, int, int],
        embedding_dim: int,
        n_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.linear_c1 = SegFormerMLP(in_channels[0], embedding_dim)
        self.linear_c2 = SegFormerMLP(in_channels[1], embedding_dim)
        self.linear_c3 = SegFormerMLP(in_channels[2], embedding_dim)
        self.linear_c4 = SegFormerMLP(in_channels[3], embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout)
        self.linear_pred = nn.Conv2d(embedding_dim, n_classes, kernel_size=1)

    def forward(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        c1, c2, c3, c4 = features
        batch_size = c4.shape[0]
        height, width = c1.shape[-2:]

        c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(batch_size, -1, c1.shape[2], c1.shape[3])
        c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(batch_size, -1, c2.shape[2], c2.shape[3])
        c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(batch_size, -1, c3.shape[2], c3.shape[3])
        c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(batch_size, -1, c4.shape[2], c4.shape[3])

        c2 = F.interpolate(c2, size=(height, width), mode="bilinear", align_corners=False)
        c3 = F.interpolate(c3, size=(height, width), mode="bilinear", align_corners=False)
        c4 = F.interpolate(c4, size=(height, width), mode="bilinear", align_corners=False)

        x = torch.cat([c4, c3, c2, c1], dim=1)
        x = self.linear_fuse(x)
        x = self.dropout(x)
        return self.linear_pred(x)


class Model(nn.Module):
    def __init__(self, in_channels: int = 3, n_classes: int = 19, dropout: float = 0.1):
        super().__init__()

        self.in_channels = in_channels

        if MODEL_CONFIG["encoder_type"] == "single_stage":
            config = MODEL_CONFIG["single_stage"]
            self.encoder = SingleStageTransformerEncoder(
                in_channels=in_channels,
                embed_dim=config["embed_dim"],
                depth=config["depth"],
                num_heads=config["num_heads"],
                patch_size=config["patch_size"],
                stride=config["stride"],
                attention_type=config["attention_type"],
                ffn_type=config["ffn_type"],
                patch_embed_type=config["patch_embed_type"],
                mlp_ratio=config["mlp_ratio"],
                sr_ratio=config["sr_ratio"],
                qkv_bias=True,
                attn_drop=config["attn_drop"],
                proj_drop=config["proj_drop"],
                ffn_drop=config["ffn_drop"],
                drop_path_rate=config["drop_path_rate"],
            )
            head_in_channels = config["embed_dim"]
        elif MODEL_CONFIG["encoder_type"] == "hierarchical":
            config = MODEL_CONFIG["hierarchical"]
            self.encoder = MixVisionTransformerEncoder(
                in_channels=in_channels,
                embed_dims=config["embed_dims"],
                depths=config["depths"],
                num_heads=config["num_heads"],
                sr_ratios=config["sr_ratios"],
                mlp_ratios=config["mlp_ratios"],
                drop_path_rate=config["drop_path_rate"],
                qkv_bias=True,
                attn_drop=config["attn_drop"],
                proj_drop=config["proj_drop"],
                ffn_drop=config["ffn_drop"],
            )
            head_in_channels = config["embed_dims"][-1]
        else:
            raise ValueError(f"Unsupported encoder type: {MODEL_CONFIG['encoder_type']}")

        if MODEL_CONFIG["decoder_type"] == "simple":
            self.decode_head = SimpleDecoderHead(head_in_channels, n_classes)
        elif MODEL_CONFIG["decoder_type"] == "segformer":
            config = MODEL_CONFIG["hierarchical"]
            self.decode_head = SegFormerHead(
                in_channels=config["embed_dims"],
                embedding_dim=config["decoder_embedding_dim"],
                n_classes=n_classes,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported decoder type: {MODEL_CONFIG['decoder_type']}")

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, but got {x.shape[1]}")

        input_size = x.shape[-2:]
        features = self.encoder(x)
        logits = self.decode_head(features)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits
