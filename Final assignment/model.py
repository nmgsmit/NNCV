import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
	"""DDRNet-23-slim style dual-resolution segmentation model.

	During training the model returns (main_logits, aux_logits).
	During evaluation it returns only main_logits to keep inference API simple.
	"""

	def __init__(self, in_channels=3, n_classes=19):
		super().__init__()
		self.in_channels = in_channels
		self.n_classes = n_classes

		# Stem: output stride 4.
		self.stem = nn.Sequential(
			ConvBNReLU(in_channels, 32, kernel_size=3, stride=2, padding=1),
			ConvBNReLU(32, 32, kernel_size=3, stride=2, padding=1),
		)

		# Shared shallow stages.
		self.layer1 = self._make_layer(32, 32, blocks=2, stride=1)
		self.layer2 = self._make_layer(32, 64, blocks=2, stride=2)  # 1/8

		# Dual-resolution stage 3.
		self.high3 = self._make_layer(64, 64, blocks=2, stride=1)
		self.down3 = ConvBNReLU(64, 128, kernel_size=3, stride=2, padding=1)  # 1/16
		self.low3 = self._make_layer(128, 128, blocks=2, stride=1)
		self.low3_to_high = nn.Conv2d(128, 64, kernel_size=1, bias=False)
		self.high3_to_low = ConvBNReLU(64, 128, kernel_size=3, stride=2, padding=1)

		# Dual-resolution stage 4.
		self.high4 = self._make_layer(64, 64, blocks=2, stride=1)
		self.down4 = ConvBNReLU(128, 256, kernel_size=3, stride=2, padding=1)  # 1/32
		self.low4 = self._make_layer(256, 256, blocks=2, stride=1)
		self.low4_to_high = nn.Conv2d(256, 64, kernel_size=1, bias=False)
		self.high4_to_low = ConvBNReLU(64, 256, kernel_size=3, stride=2, padding=1)

		# Context aggregation and heads.
		self.dappm = DAPPM(in_channels=256, branch_channels=64, out_channels=128)
		self.fuse = ConvBNReLU(64 + 128, 128, kernel_size=3, stride=1, padding=1)
		self.head = SegHead(128, 64, n_classes)
		self.aux_head = SegHead(64, 32, n_classes)

	def _make_layer(self, in_channels, out_channels, blocks, stride):
		layers = [BasicBlock(in_channels, out_channels, stride=stride)]
		for _ in range(1, blocks):
			layers.append(BasicBlock(out_channels, out_channels, stride=1))
		return nn.Sequential(*layers)

	def forward(self, x):
		if x.shape[1] != self.in_channels:
			raise ValueError(f"Expected {self.in_channels} channels, got {x.shape[1]}")

		input_size = x.shape[-2:]

		x = self.stem(x)
		x = self.layer1(x)
		x = self.layer2(x)

		# Stage 3 bilateral fusion.
		high = self.high3(x)
		low = self.low3(self.down3(x))

		high = high + F.interpolate(self.low3_to_high(low), size=high.shape[-2:], mode="bilinear", align_corners=False)
		low = low + self.high3_to_low(high)

		# Stage 4 bilateral fusion.
		high = self.high4(high)
		low = self.low4(self.down4(low))

		high = high + F.interpolate(self.low4_to_high(low), size=high.shape[-2:], mode="bilinear", align_corners=False)
		low = low + self.high4_to_low(high)

		aux_logits = self.aux_head(high)

		low_ctx = self.dappm(low)
		low_ctx = F.interpolate(low_ctx, size=high.shape[-2:], mode="bilinear", align_corners=False)

		fused = self.fuse(torch.cat([high, low_ctx], dim=1))
		main_logits = self.head(fused)

		main_logits = F.interpolate(main_logits, size=input_size, mode="bilinear", align_corners=False)
		aux_logits = F.interpolate(aux_logits, size=input_size, mode="bilinear", align_corners=False)

		if self.training:
			return main_logits, aux_logits
		return main_logits


class ConvBNReLU(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		return self.block(x)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_channels, out_channels, stride=1):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels)

		if stride != 1 or in_channels != out_channels:
			self.downsample = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(out_channels),
			)
		else:
			self.downsample = None

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out = out + identity
		out = self.relu(out)
		return out


class DAPPM(nn.Module):
	def __init__(self, in_channels, branch_channels, out_channels):
		super().__init__()
		self.scale0 = nn.Sequential(
			nn.BatchNorm2d(in_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
		)
		self.scale1 = nn.Sequential(
			nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
			nn.BatchNorm2d(in_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
		)
		self.scale2 = nn.Sequential(
			nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
			nn.BatchNorm2d(in_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
		)
		self.scale3 = nn.Sequential(
			nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
			nn.BatchNorm2d(in_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
		)
		self.scale4 = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.BatchNorm2d(in_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
		)

		self.process1 = ConvBNReLU(branch_channels, branch_channels, kernel_size=3, stride=1, padding=1)
		self.process2 = ConvBNReLU(branch_channels, branch_channels, kernel_size=3, stride=1, padding=1)
		self.process3 = ConvBNReLU(branch_channels, branch_channels, kernel_size=3, stride=1, padding=1)
		self.process4 = ConvBNReLU(branch_channels, branch_channels, kernel_size=3, stride=1, padding=1)
		self.compression = ConvBNReLU(branch_channels * 5, out_channels, kernel_size=1, stride=1, padding=0)
		self.shortcut = ConvBNReLU(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

	def forward(self, x):
		height, width = x.shape[-2:]

		x0 = self.scale0(x)
		x1 = self.process1(F.interpolate(self.scale1(x), size=(height, width), mode="bilinear", align_corners=False) + x0)
		x2 = self.process2(F.interpolate(self.scale2(x), size=(height, width), mode="bilinear", align_corners=False) + x1)
		x3 = self.process3(F.interpolate(self.scale3(x), size=(height, width), mode="bilinear", align_corners=False) + x2)
		x4 = self.process4(F.interpolate(self.scale4(x), size=(height, width), mode="bilinear", align_corners=False) + x3)

		out = self.compression(torch.cat([x0, x1, x2, x3, x4], dim=1))
		out = out + self.shortcut(x)
		return out


class SegHead(nn.Module):
	def __init__(self, in_channels, mid_channels, out_channels):
		super().__init__()
		self.block = nn.Sequential(
			ConvBNReLU(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
			nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=True),
		)

	def forward(self, x):
		return self.block(x)
