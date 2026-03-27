# SegFormer Branch Notes

This document stores the branch-by-branch explanation of the SegFormer progression for project administration and report planning, together with paper sources for the main improvements.

## v0

Branch: shared training scaffold, no separate dedicated branch.

This is the common experimental foundation used to compare the later models fairly. It contains the Cityscapes semantic segmentation setup, the general training and validation loop, checkpoint selection based on mean Dice, early stopping when mean Dice no longer improves, and experiment tracking with Weights & Biases. The point of this stage is not to introduce a special architecture, but to create a stable baseline framework so later gains can be attributed to model or recipe changes rather than to a different trainer.

Relevant sources: [Cityscapes: The Cityscapes Dataset for Semantic Urban Scene Understanding](https://arxiv.org/abs/1604.01685)

## v1

Branch: `codex/segformer-step1-plain-transformer`

This is the pure starting point: a plain transformer segmentation model with as few SegFormer-specific ideas as possible. It is mainly useful as a reference baseline, because it tells you how much performance comes purely from using a transformer before any vision-specific refinements are added.

Relevant sources: [Attention Is All You Need](https://arxiv.org/abs/1706.03762), [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

## v2

Branch: `codex/segformer-step2-overlap-patch-embedding`

This branch introduces overlap patch embedding. The purpose is to make the tokenization less harsh than standard non-overlapping patch extraction, which helps preserve local continuity and improves spatial coherence, especially around object boundaries.

Relevant sources: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

## v3

Branch: `codex/segformer-step3-efficient-attention`

This branch adds efficient self-attention with spatial reduction. The main motivation is computational efficiency: dense prediction tasks are expensive for vanilla attention, so reducing the key/value spatial resolution lets the model keep global context while remaining more practical.

Relevant sources: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

## v4

Branch: `codex/segformer-step4-mix-ffn`

This branch replaces the plain FFN with Mix-FFN. This is one of the most important steps toward SegFormer, because it adds local spatial inductive bias inside the transformer block itself through depthwise convolution, which tends to help segmentation substantially.

Relevant sources: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

## v5

Branch: `codex/segformer-step5-mit-hierarchy`

This branch introduces the hierarchical Mix Vision Transformer encoder. Instead of a single-scale transformer, the model now produces multiple feature levels at progressively lower resolutions, which is much more appropriate for semantic segmentation.

Relevant sources: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

## v6

Branch: `codex/segformer-step6-segformer-head`

This branch adds the SegFormer decode head on top of the MiT encoder. At this point the model becomes a full core SegFormer-style architecture, because the multi-scale encoder features are projected, aligned, concatenated, and fused in the lightweight decoder.

Relevant sources: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

## v7

Branch: `codex/segformer-step7-pretrained`

This branch adds pretrained MiT initialization and support for both `b0` and `b5`. The architecture is mostly the same as before, but the practical training behavior improves because the encoder no longer starts from scratch.

Relevant sources: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

## v7.1

Branch: `codex/segformer-step7-1-head-lr-lovasz`

This branch is still lightweight in resolution, but improves optimization. The decode head gets a `10x` learning-rate multiplier and the loss becomes `CrossEntropy + Lovasz-Softmax`, which is intended to better optimize segmentation overlap quality and make training more effective without making the model itself heavier.

Relevant sources: [Lovasz-Softmax: A Tractable Surrogate for the Optimization of the Intersection-Over-Union Measure in Neural Networks](https://arxiv.org/abs/1705.08790), [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

## v7.2

Branch: `codex/segformer-step7-2-msfe-fpn-syncbn-lovasz`

This branch upgrades the decoder substantially. The simple SegFormer head is replaced by an MSFE-FPN-style decoder with pyramid pooling, top-down feature fusion, attention-guided multi-scale aggregation, and `SyncBN`. This is a meaningful architecture improvement over vanilla SegFormer, especially for boundaries and small structures.

Relevant sources: [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144), [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105), [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

## v8

Branch: `codex/segformer-step8-official-cityscapes-pipeline`

This branch keeps the stronger v7.2 model and adds the official-style Cityscapes input pipeline. It is designed to be much closer to a paper-like training setup, using full-resolution base sizing, random scale augmentation, `1024x1024` crops, class-balanced crop selection, and matching full-resolution preprocessing at inference. In its current edited form, it also reuses most of the original validation split for training: the full training split is combined with the validation split except for 4 fixed Tubingen holdout images, which are kept only for monitoring and qualitative W\&B logging. It is stronger, but also much heavier.

Relevant sources: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203), [Cityscapes: The Cityscapes Dataset for Semantic Urban Scene Understanding](https://arxiv.org/abs/1604.01685)

## v9

Branch: `codex/segformer-step9-multiscale-tta`

This branch is an inference-time improvement branch. It adds multi-scale test-time augmentation in `predict.py`, averaging predictions over multiple scales and flipped inputs. The goal is not to change training, but to raise peak inference performance.

Relevant sources: [When and Why Test-Time Augmentation Works](https://arxiv.org/abs/2011.11156)

## v10

Branch: `codex/segformer-step10-segfix-refinement`

This branch builds on v9 by adding SegFix-style postprocessing in `predict.py`. The goal is to refine boundaries after prediction, improving contour quality and reducing boundary mistakes. In this implementation it is an inference-only SegFix-style approximation rather than a full paper-faithful SegFix pipeline with learned boundary and direction outputs.

Relevant sources: [SegFix: Model-Agnostic Boundary Refinement for Segmentation](https://arxiv.org/abs/2007.04269)

## Robustness Branch

Branch: `codex/segformer-robust`

This side branch builds on v7.2 and focuses on domain robustness rather than pure peak performance. It keeps the stronger decoder and optimization setup from v7.2, but augments training with synthetic weather and appearance perturbations such as artificial fog, rain, snow, low-light conditions, shadowing, and broader color or style shifts. The main goal is to help the model generalize better to harder urban scenes, different cities, worse lighting, and degraded weather conditions.

Relevant sources: [Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://arxiv.org/abs/1903.12261), [AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty](https://arxiv.org/abs/1912.02781), [FogAdapt: Self-Supervised Domain Adaptation for Semantic Segmentation of Foggy Images](https://arxiv.org/abs/2201.02588)

## Robustness Branch v2

Branch: `codex/segformer-robust-v2`

This branch builds directly on the robust SegFormer variant and combines it with the stronger peak-performance pipeline. In training, it keeps the original robust augmentations for fog, rain, snow, low-light conditions, shadowing, and broader appearance shifts, but replaces the lighter resize-based pipeline with the official Cityscapes-style setup: full-resolution base sizing, random scale augmentation, class-balanced `1024x1024` crops, and full-resolution preprocessing for validation. In its current edited form, it also follows the same data-usage pattern as the edited `v8` branch: the full training split is combined with the validation split except for 4 fixed Tubingen holdout images, which are kept only for monitoring and qualitative W\&B logging. In inference, it goes beyond the first robustness branch by adding multi-scale test-time augmentation and SegFix-style boundary refinement, making it the most complete robust SegFormer variant in the repository.

In short, `segformer-robust-v2` is the branch that combines:

- the robustness-oriented weather and domain-shift augmentations from `codex/segformer-robust`
- the official full-resolution Cityscapes training pipeline from `codex/segformer-step8-official-cityscapes-pipeline`
- the TTA and SegFix-style inference improvements from `codex/segformer-step9-multiscale-tta` and `codex/segformer-step10-segfix-refinement`

Relevant sources: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203), [Cityscapes: The Cityscapes Dataset for Semantic Urban Scene Understanding](https://arxiv.org/abs/1604.01685), [Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://arxiv.org/abs/1903.12261), [AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty](https://arxiv.org/abs/1912.02781), [FogAdapt: Self-Supervised Domain Adaptation for Semantic Segmentation of Foggy Images](https://arxiv.org/abs/2201.02588), [When and Why Test-Time Augmentation Works](https://arxiv.org/abs/2011.11156), [SegFix: Model-Agnostic Boundary Refinement for Segmentation](https://arxiv.org/abs/2007.04269)

## Efficiency Branch

Branch: `codex/segformer-efficient-b0-segfix`

This is the practical lightweight branch. It builds on v7.2, defaults to MiT-b0, keeps the stronger decoder and loss setup, allows configurable input size, and uses SegFix-style postprocessing. It is the branch to use when you want a strong but affordable model rather than the heaviest peak-performance setup.

For this efficient branch, the two useful sizes are:

- `512x1024`: the standard comparison size
- `384x768`: the smaller efficient alternative

Relevant sources: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203), [SegFix: Model-Agnostic Boundary Refinement for Segmentation](https://arxiv.org/abs/2007.04269)
