import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
	"""SegFormer-style semantic segmentation model (MiT encoder + MLP decode head).

	During training the model returns (main_logits, aux_logits).
	During evaluation it returns only main_logits to keep inference API simple.
	"""

	def __init__(self, in_channels=3, n_classes=19, embed_dims=(64, 128, 320, 512), depths=(3, 6, 40, 3), sr_ratios=(8, 4, 2, 1), num_heads=(1, 2, 5, 8), decoder_embedding_dim=768, dropout=0.1):
		super().__init__()
		self.in_channels = in_channels
		self.n_classes = n_classes

		if len(embed_dims) != 4 or len(depths) != 4 or len(sr_ratios) != 4 or len(num_heads) != 4:
			raise ValueError("embed_dims, depths, sr_ratios and num_heads must all have length 4")

		self.encoder = MixVisionTransformerEncoder(
			in_channels=in_channels,
			embed_dims=embed_dims,
			depths=depths,
			num_heads=num_heads,
			mlp_ratio=4.0,
			sr_ratios=sr_ratios,
			drop_rate=dropout,
		)

		self.decode_head = SegFormerHead(
			in_channels=embed_dims,
			embedding_dim=decoder_embedding_dim,
			n_classes=n_classes,
			dropout=dropout,
		)
		self.aux_head = nn.Conv2d(embed_dims[2], n_classes, kernel_size=1)

	def forward(self, x):
		if x.shape[1] != self.in_channels:
			raise ValueError(f"Expected {self.in_channels} channels, got {x.shape[1]}")

		input_size = x.shape[-2:]
		features = self.encoder(x)
		main_logits = self.decode_head(features)
		aux_logits = self.aux_head(features[2])

		main_logits = F.interpolate(main_logits, size=input_size, mode="bilinear", align_corners=False)
		aux_logits = F.interpolate(aux_logits, size=input_size, mode="bilinear", align_corners=False)

		if self.training:
			return main_logits, aux_logits
		return main_logits


class OverlapPatchEmbed(nn.Module):
	def __init__(self, in_channels, embed_dim, patch_size, stride):
		super().__init__()
		padding = patch_size // 2
		self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
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
		if dim % num_heads != 0:
			raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

		self.dim = dim
		self.num_heads = num_heads
		self.head_dim = dim // num_heads
		self.scale = self.head_dim ** -0.5
		self.sr_ratio = sr_ratio

		self.q = nn.Linear(dim, dim)
		self.kv = nn.Linear(dim, dim * 2)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

		if sr_ratio > 1:
			self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
			self.norm = nn.LayerNorm(dim)
		else:
			self.sr = None
			self.norm = None

	def forward(self, x, h, w):
		b, n, c = x.shape

		q = self.q(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

		if self.sr is not None:
			x_ = x.transpose(1, 2).reshape(b, c, h, w)
			x_ = self.sr(x_).reshape(b, c, -1).transpose(1, 2)
			x_ = self.norm(x_)
		else:
			x_ = x

		kv = self.kv(x_).reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
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
	def __init__(self, dim, num_heads, mlp_ratio=4.0, sr_ratio=1, drop=0.0):
		super().__init__()
		self.norm1 = nn.LayerNorm(dim)
		self.attn = EfficientSelfAttention(dim=dim, num_heads=num_heads, sr_ratio=sr_ratio, attn_drop=drop, proj_drop=drop)
		self.norm2 = nn.LayerNorm(dim)
		hidden_dim = int(dim * mlp_ratio)
		self.mlp = MixFFN(dim=dim, hidden_dim=hidden_dim, drop=drop)

	def forward(self, x, h, w):
		x = x + self.attn(self.norm1(x), h, w)
		x = x + self.mlp(self.norm2(x), h, w)
		return x


class MixVisionStage(nn.Module):
	def __init__(self, in_channels, embed_dim, depth, num_heads, sr_ratio, patch_size, stride, drop_rate=0.0):
		super().__init__()
		self.patch_embed = OverlapPatchEmbed(in_channels=in_channels, embed_dim=embed_dim, patch_size=patch_size, stride=stride)
		self.blocks = nn.ModuleList(
			[TransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0, sr_ratio=sr_ratio, drop=drop_rate) for _ in range(depth)]
		)
		self.norm = nn.LayerNorm(embed_dim)

	def forward(self, x):
		tokens, h, w = self.patch_embed(x)
		for block in self.blocks:
			tokens = block(tokens, h, w)
		tokens = self.norm(tokens)
		b, _, c = tokens.shape
		out = tokens.transpose(1, 2).reshape(b, c, h, w)
		return out


class MixVisionTransformerEncoder(nn.Module):
	def __init__(self, in_channels, embed_dims, depths, num_heads, sr_ratios, mlp_ratio=4.0, drop_rate=0.0):
		super().__init__()
		self.mlp_ratio = mlp_ratio
		self.stage1 = MixVisionStage(
			in_channels=in_channels,
			embed_dim=embed_dims[0],
			depth=depths[0],
			num_heads=num_heads[0],
			sr_ratio=sr_ratios[0],
			patch_size=7,
			stride=4,
			drop_rate=drop_rate,
		)
		self.stage2 = MixVisionStage(
			in_channels=embed_dims[0],
			embed_dim=embed_dims[1],
			depth=depths[1],
			num_heads=num_heads[1],
			sr_ratio=sr_ratios[1],
			patch_size=3,
			stride=2,
			drop_rate=drop_rate,
		)
		self.stage3 = MixVisionStage(
			in_channels=embed_dims[1],
			embed_dim=embed_dims[2],
			depth=depths[2],
			num_heads=num_heads[2],
			sr_ratio=sr_ratios[2],
			patch_size=3,
			stride=2,
			drop_rate=drop_rate,
		)
		self.stage4 = MixVisionStage(
			in_channels=embed_dims[2],
			embed_dim=embed_dims[3],
			depth=depths[3],
			num_heads=num_heads[3],
			sr_ratio=sr_ratios[3],
			patch_size=3,
			stride=2,
			drop_rate=drop_rate,
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
		self.proj1 = nn.Conv2d(in_channels[0], embedding_dim, kernel_size=1)
		self.proj2 = nn.Conv2d(in_channels[1], embedding_dim, kernel_size=1)
		self.proj3 = nn.Conv2d(in_channels[2], embedding_dim, kernel_size=1)
		self.proj4 = nn.Conv2d(in_channels[3], embedding_dim, kernel_size=1)

		self.fuse = nn.Sequential(
			nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False),
			nn.BatchNorm2d(embedding_dim),
			nn.ReLU(inplace=True),
			nn.Dropout2d(dropout),
		)
		self.classifier = nn.Conv2d(embedding_dim, n_classes, kernel_size=1)

	def forward(self, features):
		c1, c2, c3, c4 = features
		h, w = c1.shape[-2:]

		p1 = self.proj1(c1)
		p2 = F.interpolate(self.proj2(c2), size=(h, w), mode="bilinear", align_corners=False)
		p3 = F.interpolate(self.proj3(c3), size=(h, w), mode="bilinear", align_corners=False)
		p4 = F.interpolate(self.proj4(c4), size=(h, w), mode="bilinear", align_corners=False)

		fused = self.fuse(torch.cat([p1, p2, p3, p4], dim=1))
		return self.classifier(fused)
