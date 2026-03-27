# Final Assignment: Cityscape Challenge  

Welcome to the **Cityscape Challenge**, the final project for this course!  

In this assignment, you'll put your knowledge of Neural Networks (NNs) for computer vision into action by tackling real-world problems using the **CityScapes dataset**. This dataset contains large-scale, high-quality images of urban environments, making it perfect for tasks like **semantic segmentation** and **object detection**.  

This challenge is designed to push your skills further, focusing on practical and often under-explored issues crucial for deploying computer vision models in real-world scenarios.  

---

## Benchmarks  

The competition comprises four benchmarks, each targeting a specific aspect of model performance:  

1. **Peak performance**  
   This benchmark evaluates your model's segmentation accuracy on a clean, standardized test set. Your goal is to achieve the highest segmentation scores here. **Everyone should submit a model to this benchmark optimized for maximum performance**. However, it's crucial to implement changes thoughtfully and be able to justify them in your research paper. Ultimately, the focus should be on the scientific contributions of your adaptations rather than solely aiming for the highest score.

The following benchmarks 2–4 are optional, and **you should select one** to compare against the Peak Performance benchmark. This allows you to analyze how your model performs under different conditions and gain deeper insights beyond just optimizing for the highest score.

2. **Robustness**  
   This benchmark tests how well your model performs under challenging conditions, such as changes in lighting, weather, or image quality. Consistency is key in this category.  

3. **Efficiency**  
   Practical applications often require compact models. This benchmark emphasizes creating smaller models that maintain acceptable performance. It’s particularly relevant for edge devices where large models are infeasible.  

4. **Out-of-distribution detection**  
   Models often encounter data that differs from the training distribution, leading to unreliable predictions. This benchmark evaluates your model's ability to detect and handle such out-of-distribution samples.  

> **IMPORTANT NOTE**: The **Peak Permomance** benchmark will also serve as the baseline server, and all participants must submit a baseline model here. This means that you can just train the already provided model in the repo. The training code for this model is also already provided. The baseline submission serves two purposes: ensuring that everyone is familiar with working on an HPC cluster and providing a reference point for evaluating the impact of different adaptations in your other benchmark submissions (you need to show these improvements compared to the baseline in your report!). The Baseline benchmark will close on **Tuesday, March 17, at 11:59 P.M. (GMT+1)**. To avoid last-minute issues, start preparing your submission early. This will also give you time to ask questions during the scheduled computer classes if needed.

---

## Deliverables  

Your final submission will consist of the following:  

### 1. Research paper  
Write a **3-4 page research paper** in [IEEE double-column format](https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn), addressing (at least) the following:  

- **Abstract**: Summarize the current problems, your key steps for addressing them and your main findings in about 100-300 words.
- **Introduction**: Present the problem, challenges, and potential solutions based on existing literature.  
- **Methods**: Describe your dataset(s), outline the baseline approach using an off-the-shelf segmentation model and define the enhancements you made for the specific benchmarks you participated.  
- **Results**: Show and describe your results based on performance metrics and examples. Use figures and tables to support your findings. 
- **Discussion**: Discuss the impact and potential of your main findings. Also discuss limitations and suggest future improvements.

> **Submission**: Submit your paper as a PDF document via **Canvas**.

The paper will be graded based on clarity, experimental design, insight, and originality.  

### 2. Code repository  
Push all relevant code to a **public GitHub repository** with a README.md file detailing:  
- Required libraries and installation instructions.  
- Steps to run your code.  
- Your Codalab username and TU/e email address for correct mapping across systems.  

### 3. Challenge platform submissions  
The Cityscape Challenge will be hosted on a **dedicated course compute platform** (instead of Codalab used in previous years).

You will receive clear, step-by-step instructions for making submission once the final assignment begins.

---

## Grading and Bonus Points  

The final assignment accounts for **50% of your course grade**. Additionally, bonus points are available:  

- **Top 3 in any benchmark**: +0.25 to your final assignment grade.  
- **Best performance in any benchmark**: +0.5 to your final assignment grade.  

For example, achieving the best performance in 'Peak Performance' and a top 3 spot in another benchmark will earn you a 0.75 bonus.  

> **Note**: The bonus is optional. A great report with an innovative solution that doesn't rank highly can still earn a perfect score (10).  

---

## Important Notes  

- Ensure a proper **train-validation split** of the CityScapes dataset.  
- Training your model may take multiple hours; plan accordingly.  
- Use ideas from literature but remember to **cite all sources**. Plagiarism will not be tolerated.  
- For questions or challenges, use the **Discussions** section of this repository to collaborate with peers.  

## SegFormer Branch Progression

The repository also contains a branch-by-branch SegFormer progression that starts from a minimal transformer segmentation model and gradually builds toward stronger architecture, training, inference, robustness, and efficiency variants. The purpose of this progression is to keep the development history scientifically interpretable: each branch adds a clearly motivated component or recipe change on top of the previous one.

### v0

Branch: shared training scaffold, no separate dedicated branch.

This is the common experimental foundation used to compare the later models fairly. It contains the Cityscapes semantic segmentation setup, the general training and validation loop, checkpoint selection based on mean Dice, early stopping when mean Dice no longer improves, and experiment tracking with Weights & Biases. The point of this stage is not to introduce a special architecture, but to create a stable baseline framework so later gains can be attributed to model or recipe changes rather than to a different trainer.

Relevant sources: [Cityscapes: The Cityscapes Dataset for Semantic Urban Scene Understanding](https://arxiv.org/abs/1604.01685)

### v1

Branch: `codex/segformer-step1-plain-transformer`

This is the pure starting point: a plain transformer segmentation model with as few SegFormer-specific ideas as possible. It is mainly useful as a reference baseline, because it tells you how much performance comes purely from using a transformer before any vision-specific refinements are added.

Relevant sources: [Attention Is All You Need](https://arxiv.org/abs/1706.03762), [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

### v2

Branch: `codex/segformer-step2-overlap-patch-embedding`

This branch introduces overlap patch embedding. The purpose is to make the tokenization less harsh than standard non-overlapping patch extraction, which helps preserve local continuity and improves spatial coherence, especially around object boundaries.

Relevant sources: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

### v3

Branch: `codex/segformer-step3-efficient-attention`

This branch adds efficient self-attention with spatial reduction. The main motivation is computational efficiency: dense prediction tasks are expensive for vanilla attention, so reducing the key/value spatial resolution lets the model keep global context while remaining more practical.

Relevant sources: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

### v4

Branch: `codex/segformer-step4-mix-ffn`

This branch replaces the plain FFN with Mix-FFN. This is one of the most important steps toward SegFormer, because it adds local spatial inductive bias inside the transformer block itself through depthwise convolution, which tends to help segmentation substantially.

Relevant sources: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

### v5

Branch: `codex/segformer-step5-mit-hierarchy`

This branch introduces the hierarchical Mix Vision Transformer encoder. Instead of a single-scale transformer, the model now produces multiple feature levels at progressively lower resolutions, which is much more appropriate for semantic segmentation.

Relevant sources: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

### v6

Branch: `codex/segformer-step6-segformer-head`

This branch adds the SegFormer decode head on top of the MiT encoder. At this point the model becomes a full core SegFormer-style architecture, because the multi-scale encoder features are projected, aligned, concatenated, and fused in the lightweight decoder.

Relevant sources: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

### v7

Branch: `codex/segformer-step7-pretrained`

This branch adds pretrained MiT initialization and support for both `b0` and `b5`. The architecture is mostly the same as before, but the practical training behavior improves because the encoder no longer starts from scratch.

Relevant sources: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

### v7.1

Branch: `codex/segformer-step7-1-head-lr-lovasz`

This branch is still lightweight in resolution, but improves optimization. The decode head gets a `10x` learning-rate multiplier and the loss becomes `CrossEntropy + Lovasz-Softmax`, which is intended to better optimize segmentation overlap quality and make training more effective without making the model itself heavier.

Relevant sources: [Lovasz-Softmax: A Tractable Surrogate for the Optimization of the Intersection-Over-Union Measure in Neural Networks](https://arxiv.org/abs/1705.08790), [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

### v7.2

Branch: `codex/segformer-step7-2-msfe-fpn-syncbn-lovasz`

This branch upgrades the decoder substantially. The simple SegFormer head is replaced by an MSFE-FPN-style decoder with pyramid pooling, top-down feature fusion, attention-guided multi-scale aggregation, and `SyncBN`. This is a meaningful architecture improvement over vanilla SegFormer, especially for boundaries and small structures.

Relevant sources: [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144), [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105), [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

### v8

Branch: `codex/segformer-step8-official-cityscapes-pipeline`

This branch keeps the stronger v7.2 model and adds the official-style Cityscapes input pipeline. It is designed to be much closer to a paper-like training setup, using full-resolution base sizing, random scale augmentation, `1024x1024` crops, class-balanced crop selection, and matching full-resolution preprocessing at inference. It is stronger, but also much heavier.

Relevant sources: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203), [Cityscapes: The Cityscapes Dataset for Semantic Urban Scene Understanding](https://arxiv.org/abs/1604.01685)

### v9

Branch: `codex/segformer-step9-multiscale-tta`

This branch is an inference-time improvement branch. It adds multi-scale test-time augmentation in `predict.py`, averaging predictions over multiple scales and flipped inputs. The goal is not to change training, but to raise peak inference performance.

Relevant sources: [When and Why Test-Time Augmentation Works](https://arxiv.org/abs/2011.11156)

### v10

Branch: `codex/segformer-step10-segfix-refinement`

This branch builds on v9 by adding SegFix-style postprocessing in `predict.py`. The goal is to refine boundaries after prediction, improving contour quality and reducing boundary mistakes. In this implementation it is an inference-only SegFix-style approximation rather than a full paper-faithful SegFix pipeline with learned boundary and direction outputs.

Relevant sources: [SegFix: Model-Agnostic Boundary Refinement for Segmentation](https://arxiv.org/abs/2007.04269)

### Robustness Branch

Branch: `codex/segformer-robust`

This side branch builds on v7.2 and focuses on domain robustness rather than pure peak performance. It keeps the stronger decoder and optimization setup from v7.2, but augments training with synthetic weather and appearance perturbations such as artificial fog, rain, snow, low-light conditions, shadowing, and broader color or style shifts. The main goal is to help the model generalize better to harder urban scenes, different cities, worse lighting, and degraded weather conditions.

Relevant sources: [Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://arxiv.org/abs/1903.12261), [AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty](https://arxiv.org/abs/1912.02781), [FogAdapt: Self-Supervised Domain Adaptation for Semantic Segmentation of Foggy Images](https://arxiv.org/abs/2201.02588)

### Robustness Branch v2

Branch: `codex/segformer-robust-v2`

This branch builds directly on the robust SegFormer variant and combines it with the stronger peak-performance pipeline. In training, it keeps the original robust augmentations for fog, rain, snow, low-light conditions, shadowing, and broader appearance shifts, but replaces the lighter resize-based pipeline with the official Cityscapes-style setup: full-resolution base sizing, random scale augmentation, class-balanced `1024x1024` crops, and full-resolution preprocessing for validation. In inference, it goes beyond the first robustness branch by adding multi-scale test-time augmentation and SegFix-style boundary refinement, making it the most complete robust SegFormer variant in the repository.

Relevant sources: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203), [Cityscapes: The Cityscapes Dataset for Semantic Urban Scene Understanding](https://arxiv.org/abs/1604.01685), [Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://arxiv.org/abs/1903.12261), [AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty](https://arxiv.org/abs/1912.02781), [FogAdapt: Self-Supervised Domain Adaptation for Semantic Segmentation of Foggy Images](https://arxiv.org/abs/2201.02588), [When and Why Test-Time Augmentation Works](https://arxiv.org/abs/2011.11156), [SegFix: Model-Agnostic Boundary Refinement for Segmentation](https://arxiv.org/abs/2007.04269)

### Efficiency Branch

Branch: `codex/segformer-efficient-b0-segfix`

This is the practical lightweight branch. It builds on v7.2, defaults to MiT-b0, keeps the stronger decoder and loss setup, allows configurable input size, and uses SegFix-style postprocessing. It is the branch to use when a strong but affordable model is preferred over the heaviest peak-performance setup.

For this efficient branch, the two useful sizes are:

- `512x1024`: the standard comparison size
- `384x768`: the smaller efficient alternative

Relevant sources: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203), [SegFix: Model-Agnostic Boundary Refinement for Segmentation](https://arxiv.org/abs/2007.04269)

For convenience, a separate copy of these administration notes is also stored in `README-Branches.md`.

---

We wish you the best of luck in this challenge and are excited to see the innovative solutions you develop! 🚀
