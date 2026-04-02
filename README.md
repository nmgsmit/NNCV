# 5LSM0: Neural Networks for Computer Vision

Welcome to the repository for **5LSM0: Neural Networks for Computer Vision**, a course offered by the Department of Electrical Engineering at Eindhoven University of Technology. This course is hosted by the [Architectures for Relaible Image Analysis Lab](https://github.com/TUE-ARIA).

## Overview

This repository contains all the assignments and supplementary materials for the course. The weekly assignments are designed as **Jupyter Notebooks**, providing practical, hands-on experience with the concepts discussed during lectures. These notebooks will help you gain familiarity with implementing neural networks using **PyTorch** in **Python**. For the final assignment, you will apply the knowledge gained throughout the course by coding in native Python and working with a compute cluster, providing valuable experience in real-world computational environments.

### Weekly Assignments

The weekly assignments are structured to guide you through foundational and advanced topics in neural networks for computer vision. These assignments are:
- **Optional**: They are not mandatory but serve as valuable practice to build your coding skills.
- **Hands-On**: Focused on applying theoretical knowledge from the lectures into real-world implementations.

### Final Assignment

The **final assignment** is the cornerstone of this course and accounts for **50% of your final grade**. In this project, you will:
1. Work on a real-world problem using the **CityScapes dataset**.
2. Train neural networks and validate their performance against established baselines.
3. Document your results and insights in a detailed report.

This final assignment requires a deeper dive into the subject, pushing you to apply the knowledge and skills gained throughout the course.

The final assignment will start in week 3 (February 24th), once all core lectures have been completed, ensuring you have the necessary foundation to work on the project.

## Current Branch Notes

The active experimental branch for the final assignment is `codex/segformer-robust-v2`. The implementation lives in `Final assignment/` and extends the earlier `codex/segformer-robust` branch.

Main changes on `codex/segformer-robust-v2`:
- Keeps the same SegFormer backbone family and the same MSFE-FPN decoder stack from the earlier robust branch, including `MSFEFPNHead`, `PyramidPoolingModule`, `ScaleAttentionFusion`, and `SyncBatchNorm`.
- Switches training to an official Cityscapes-style full-resolution pipeline with `1024x2048` base images, random scaling, `1024x1024` class-balanced crops, and default `b5` training with batch size `1`.
- Uses the Cityscapes `train` split plus most of `val` for training, while reserving a fixed 4-image holdout from `val` for validation.
- Retains the robustness-focused weather and appearance augmentations, including fog, rain, snow, low-light, domain shift, and synthetic shadow perturbations.
- Evaluates with multi-scale test-time augmentation using scales `0.75`, `1.0`, and `1.25` with horizontal flip enabled by default.
- Adds SegFix-style boundary refinement during inference for sharper prediction boundaries.
- Improves submission robustness by adding configurable prediction settings, automatic device selection, CUDA usability checks, and CUDA OOM fallback profiles that retry with lighter inference settings when needed.

## Authors and Contact

This course material is developed and maintained by the following contributors:  

- **Cris H.B. Claessens**  
  Email: [c.h.b.claessens@tue.nl](mailto:c.h.b.claessens@tue.nl)  

- **Tim J.M. Jaspers**  
  Email: [t.j.m.jaspers@tue.nl](mailto:t.j.m.jaspers@tue.nl)

- **Francisco De Espírito Santo e Caetano**  
  Email: [f.t.de.espirito.santo.e.caetano@tue.nl](mailto:f.t.de.espirito.santo.e.caetano@tue.nl)

- **Lemar Abdi**  
  Email: [l.abdi@tue.nl](mailto:l.abdi@tue.nl)

- **dr. Christiaan G.A. Viviers**  
  Email: [c.g.a.vivers@tue.nl](mailto:c.g.a.vivers@tue.nl)

If you have questions or need assistance, you can always reach out to us via email. However, we strongly encourage you to post your questions in the **Discussions** section of this GitHub repository. This way, other students can benefit from the conversations and contribute by helping each other out.
