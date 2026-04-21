# SegFormer Branch Notes

This document stores the branch-by-branch explanation of the SegFormer progression for project administration and report planning, together with paper sources for the main improvements.

## Some models are still in development and do not contribute to the final Paper*

## Baseline v0

Branch: shared training scaffold, no separate dedicated branch.

This is the common experimental foundation used to compare the later models fairly. It contains the Cityscapes semantic segmentation setup, the general training and validation loop, checkpoint selection based on mean Dice, early stopping when mean Dice no longer improves, and experiment tracking with Weights & Biases. The point of this stage is not to introduce a special architecture, but to create a stable baseline framework so later gains can be attributed to model or recipe changes rather than to a different trainer.

Relevant sources: [Cityscapes: The Cityscapes Dataset for Semantic Urban Scene Understanding](https://arxiv.org/abs/1604.01685)

## v1

Branch: `SegformerV1_ViT`

This is the pure starting point: a plain transformer segmentation model with as few SegFormer-specific ideas as possible. It is mainly useful as a reference baseline, because it tells you how much performance comes purely from using a transformer before any vision-specific refinements are added.

Relevant sources: [Attention Is All You Need](https://arxiv.org/abs/1706.03762), [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

## v2

Branch: `SegformerV2_OverlapPatch`

This branch introduces overlap patch embedding. The purpose is to make the tokenization less harsh than standard non-overlapping patch extraction, which helps preserve local continuity and improves spatial coherence, especially around object boundaries.

Relevant sources: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

## v3

Branch: `SegformerV3_EfficientSA`

This branch adds efficient self-attention with spatial reduction. The main motivation is computational efficiency: dense prediction tasks are expensive for vanilla attention, so reducing the key/value spatial resolution lets the model keep global context while remaining more practical.

Relevant sources: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

## v4

Branch: `SegformerV4_MixFFN`

This branch replaces the plain FFN with Mix-FFN. This is one of the most important steps toward SegFormer, because it adds local spatial inductive bias inside the transformer block itself through depthwise convolution, which tends to help segmentation substantially.

Relevant sources: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

## v5

Branch: `SegformerV5_MiT`

This branch introduces the hierarchical Mix Vision Transformer encoder. Instead of a single-scale transformer, the model now produces multiple feature levels at progressively lower resolutions, which is much more appropriate for semantic segmentation.

Relevant sources: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

## v6

Branch: `SegformerV6_SegHead`

This branch adds the SegFormer decode head on top of the MiT encoder. At this point the model becomes a full core SegFormer-style architecture, because the multi-scale encoder features are projected, aligned, concatenated, and fused in the lightweight decoder.

Relevant sources: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

## v7

Branch: `SegformerV7_Pretrained`

This branch adds pretrained MiT initialization and support for both `b0` and `b5`. The architecture is mostly the same as before, but the practical training behavior improves because the encoder no longer starts from scratch.

Relevant sources: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

## v7.1

Branch: `SegformerV7.1_LovaszLoss`

This branch is still lightweight in resolution, but improves optimization. The decode head gets a `10x` learning-rate multiplier and the loss becomes `CrossEntropy + Lovasz-Softmax`, which is intended to better optimize segmentation overlap quality and make training more effective without making the model itself heavier.

Relevant sources: [Lovasz-Softmax: A Tractable Surrogate for the Optimization of the Intersection-Over-Union Measure in Neural Networks](https://arxiv.org/abs/1705.08790), [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

## v7.2

Branch: `SegformerV7.2_FPNdecoder`

This branch upgrades the decoder substantially. The simple SegFormer head is replaced by an MSFE-FPN-style decoder with pyramid pooling, top-down feature fusion, attention-guided multi-scale aggregation, and `SyncBN`. This is a meaningful architecture improvement over vanilla SegFormer, especially for boundaries and small structures.

Relevant sources: [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144), [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105), [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

## v8

Branch: `SegformerV8_HeavyTraining`

This branch keeps the stronger v7.2 model and adds the official-style Cityscapes input pipeline. It is designed to be much closer to a paper-like training setup, using full-resolution base sizing, random scale augmentation, `1024x1024` crops, class-balanced crop selection, and matching full-resolution preprocessing at inference. In its current edited form, it also reuses most of the original validation split for training: the full training split is combined with the validation split except for 4 fixed Tubingen holdout images, which are kept only for monitoring and qualitative W\&B logging. It is stronger, but also much heavier.

Relevant sources: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203), [Cityscapes: The Cityscapes Dataset for Semantic Urban Scene Understanding](https://arxiv.org/abs/1604.01685)

## v9

Branch: `SegformerV9_TTA`

This branch is an inference-time improvement branch. It adds multi-scale test-time augmentation in `predict.py`, averaging predictions over multiple scales and flipped inputs. The goal is not to change training, but to raise peak inference performance.

Relevant sources: [When and Why Test-Time Augmentation Works](https://arxiv.org/abs/2011.11156)

## v10

Branch: `SegformerV10_Segfix`

This branch builds on v9 by adding SegFix-style postprocessing in `predict.py`. The goal is to refine boundaries after prediction, improving contour quality and reducing boundary mistakes. In this implementation it is an inference-only SegFix-style approximation rather than a full paper-faithful SegFix pipeline with learned boundary and direction outputs.

Relevant sources: [SegFix: Model-Agnostic Boundary Refinement for Segmentation](https://arxiv.org/abs/2007.04269)

## Robustness Branch

Branch: `SegformerRobust`

This side branch builds on v7.2 and focuses on domain robustness rather than pure peak performance. It keeps the stronger decoder and optimization setup from v7.2, but augments training with synthetic weather and appearance perturbations such as artificial fog, rain, snow, low-light conditions, shadowing, and broader color or style shifts. The main goal is to help the model generalize better to harder urban scenes, different cities, worse lighting, and degraded weather conditions.

Relevant sources: [Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://arxiv.org/abs/1903.12261), [AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty](https://arxiv.org/abs/1912.02781), [FogAdapt: Self-Supervised Domain Adaptation for Semantic Segmentation of Foggy Images](https://arxiv.org/abs/2201.02588)

## Robustness Branch v2

Branch: `SegformerRobustV2`

This branch builds directly on the robust SegFormer variant and combines it with the stronger peak-performance pipeline. In training, it keeps the original robust augmentations for fog, rain, snow, low-light conditions, shadowing, and broader appearance shifts, but replaces the lighter resize-based pipeline with the official Cityscapes-style setup: full-resolution base sizing, random scale augmentation, class-balanced `1024x1024` crops, and full-resolution preprocessing for validation. In its current edited form, it also follows the same data-usage pattern as the edited `v8` branch: the full training split is combined with the validation split except for 4 fixed Tubingen holdout images, which are kept only for monitoring and qualitative W\&B logging. In inference, it goes beyond the first robustness branch by adding multi-scale test-time augmentation and SegFix-style boundary refinement, making it the most complete robust SegFormer variant in the repository.

In short, `SegformerRobustV2` is the branch that combines:

- the robustness-oriented weather and domain-shift augmentations from `SegformerRobust`
- the official full-resolution Cityscapes training pipeline from `SegformerV8_HeavyTraining`
- the TTA and SegFix-style inference improvements from `SegformerV9_TTA` and `SegformerV10_Segfix`

Relevant sources: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203), [Cityscapes: The Cityscapes Dataset for Semantic Urban Scene Understanding](https://arxiv.org/abs/1604.01685), [Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://arxiv.org/abs/1903.12261), [AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty](https://arxiv.org/abs/1912.02781), [FogAdapt: Self-Supervised Domain Adaptation for Semantic Segmentation of Foggy Images](https://arxiv.org/abs/2201.02588), [When and Why Test-Time Augmentation Works](https://arxiv.org/abs/2011.11156), [SegFix: Model-Agnostic Boundary Refinement for Segmentation](https://arxiv.org/abs/2007.04269)

## Efficiency Branch

Branch: `SegformerEfficientb0`

This is the practical lightweight branch. It builds on v7.2, defaults to MiT-b0, keeps the stronger decoder and loss setup, allows configurable input size, and uses SegFix-style postprocessing. It is the branch to use when you want a strong but affordable model rather than the heaviest peak-performance setup.

For this efficient branch, the two useful sizes are:

- `512x1024`: the standard comparison size
- `384x768`: the smaller efficient alternative

Relevant sources: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203), [SegFix: Model-Agnostic Boundary Refinement for Segmentation](https://arxiv.org/abs/2007.04269)

## DDRNet

Branch: `DDRNET`

This is the final lightweight DDRNet branch used for the efficiency benchmark. It keeps the dual-resolution DDRNet-style segmentation design, but compresses it into a much smaller practical variant by using roughly half-width channels throughout the network, trimming the deepest stage from two residual blocks to one, and skipping the auxiliary head during inference. The main goal is to preserve the strong real-time segmentation structure of DDRNet while reducing compute enough to make it competitive for the efficiency setting with only a moderate performance drop.

In practice, this branch is the compact DDRNet line that was developed from the earlier internal DDRNet experiments and finalized as the single submission-ready DDRNet branch. It is the branch to use when the priority is a strong FLOPs reduction rather than maximum peak SegFormer accuracy.

Relevant sources: [Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes](https://arxiv.org/abs/2101.06085), [Cityscapes: The Cityscapes Dataset for Semantic Urban Scene Understanding](https://arxiv.org/abs/1604.01685)
