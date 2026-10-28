---
layout: page
permalink: /press/vector-icml-2026/
title: Vector researchers advance generative AI, responsible AI, and scientific discovery at ICML 2026
description: Archived excerpt from a Vector Institute announcement
nav: false
sitemap: false
---

{% include press_archive_note.liquid
  excerpt=true
  description="a <a href='https://vectorinstitute.ai'>Vector Institute</a> announcement"
  date="July 2, 2026"
  url="https://vectorinstitute.ai/vector-researchers-icml-2026-accepted-papers/"
%}

{% include figure.liquid path="assets/img/press/vector_icml_2026_header.jpg" alt="Vector researchers at ICML 2026" class="img-fluid rounded z-depth-1" sizes="(min-width: 930px) 900px, 95vw" %}

Vector researchers are presenting work across a broad front at this year’s International Conference on Machine Learning (ICML), taking place July 6–11, 2026 in Seoul, South Korea. With 73 accepted papers – Vector’s strongest showing at ICML to date – and 11 spotlight papers among them, Vector Faculty Members, Vector Faculty Affiliates, Vector Distinguished Postdoctoral Fellows, and Vector staff are contributing to some of the most active areas of machine learning research today. […]

### [SparseOpt: Addressing Normalization-induced Gradient Skew in Sparse Training](https://openreview.net/attachment?id=o9JP7N8YO9&name=pdf)

Adnan Mohammed, Rohan Jain, Tom Jacobs, Ekansh Sharma, Rahul G. Krishnan (Vector Faculty Member), Rebekka Burkholz, Yani Ioannou

#### Abstract

Dynamic Sparse Training (DST) methods train neural networks by maintaining sparsity while dynamically adapting the network topology. Despite the promise of reduced computation, DST methods converge significantly slower than dense training, often requiring comparable training time to achieve similar accuracy. We demonstrate both analytically and empirically that Batch Normalization (BN) adversely affects sparse training, and propose SparseOpt — a sparsity-aware optimizer — to address this. Experiments on ResNet models across CIFAR-100 and ImageNet demonstrate consistently faster convergence and improved generalization with our proposed method. Our work highlights the limitations of current normalization layers in sparse training and provides the first systematic study of the interaction between Batch Normalization, sparse layers, and DST, taking a significant step toward making DST practically competitive with dense training.

#### Summary

Sparse training aims to train neural networks using only a small fraction of the connections, instead of using every parameter in the network. Since fewer connections are active, sparse training has the potential to significantly reduce the computational and memory cost of training large models. However, in practice, sparse neural networks are much harder to train than standard dense networks. Existing sparse training methods often converge substantially slower and may require significantly longer training schedules to match the accuracy of dense models, limiting their practical efficiency benefits.

To stabilize and accelerate the training of deep neural networks, modern architectures almost universally rely on normalization layers such as Batch Normalization (BN). Intuitively, BN keeps activations and gradients well-scaled during training, which helps optimization remain stable and enables faster convergence. While BN has been extensively studied in dense neural networks, prior work largely assumed that it behaves similarly in sparse networks. In this work, we show both theoretically and empirically that this assumption does not hold. Specifically, we demonstrate that BN interacts poorly with heterogeneous sparse connectivity, leading to neuron-dependent gradient scaling that distorts optimization dynamics and destabilizes Dynamic Sparse Training (DST).

Motivated by this observation, we propose a simple sparsity-aware correction method, SparseOpt, that accounts for the gradient imbalance introduced by BN. Our method improves optimization stability, accelerates convergence, and consistently improves sparse training performance across multiple datasets and architectures.

**TLDR:** Batch Normalization adversely affects sparse training; we proposed a sparsity-aware optimization method to mitigate this and improves training convergence and generalization.
