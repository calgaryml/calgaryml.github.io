---
layout: distill
title: "Dynamic Sparse Training with Structured Sparsity"
description: "Learning Performant and Efficient Representations suitable for Hardware Acceleration"
tags: dynamic sparse training structured sparsity compression RigL SET
toc: true
related_posts: true
pretty_table: true
date: 2024-11-14
featured: false
authors:
  - name: Mike Lasby
    url: "https://github.com/mklasby"
    affiliations:
      name: University of Calgary
  - name: Anna Golubeva
    url: "https://annagolubeva.github.io"
    affiliations:
      name: MIT, IAIFI
  - name: Utku Evci
    url: "https://utkuevci.com"
    affiliations:
      name: Google DeepMind
  - name: Mihai Nica
    url: "https://nicam.uoguelph.ca"
    affiliations:
      name: University of Guelph, Vector Institute for AI
  - name: Yani Ioannou
    url: "https://yani.ai"
    affiliations:
      name: University of Calgary
bibliography: 2025-05-17-training-sparse-structured-nn.bib

doi: 10.48550/arXiv.2305.02299
paper_url: https://openreview.net/pdf?id=kOBkxFRKTA

_styles: >
  .MJX-TEX,
  .MJX-TEX * {
      font-family: "Times New Roman";
  }
---

## TL;DR

The challenge in training sparse neural networks is to achieve both high accuracy and practical hardware acceleration. Unstructured sparsity often yields good performance but is hard to speed up, while traditional structured sparsity can hurt performance.

Our International Conference in Learning Representations (ICLR) 2024 paper, ["Dynamic Sparse Training with Structured Sparsity"](https://openreview.net/forum?id=kOBkxFRKTA) <d-cite key="Lasby2024SRigL"></d-cite> introduces Structured RigL (SRigL) to addresses this by dynamically learning hardware-friendly sparse weight representations without sacrificing accuracy. Key findings of the work include:

- SRigL successfully learns a combination of fine-grained N:M structured sparsity (constant fan-in) and neuron-level sparsity (neuron ablation) dynamically from a sparse initialization.
- The explicit integration of neuron ablation, a behavior implicitly learned by unstructured DST methods at high sparsities, is crucial for SRigL to match the generalization performance of dense and unstructured sparse models, even at extreme sparsities (up to 99%).
- The learned structured sparsity enables a "condensed sparse representation," which translates to significant real-world inference speedups on commodity CPUs (up to $3.4 \times$ vs. dense at 90% sparsity) and GPUs (up to $1.7 \times$ vs. dense at 90% sparsity), outperforming unstructured sparse formats in many practical scenarios.
- SRigL demonstrates a viable path to train sparse neural networks that are both highly accurate and practically efficient, bridging the gap between unstructured DST performance and structured sparsity acceleration.

## The Quest for More Efficient Neural Networks

State-of-the-art deep neural networks (DNNs) have achieved remarkable feats, but their ever-increasing size brings ballooning training costs, often outstripping Moore's Law. This trend makes cutting-edge AI research less accessible. While techniques exist to prune trained dense models, effectively reducing their parameter count by 85-95% without sacrificing generalization, they relay on dense pre-training. Can we train these sparse, efficient networks without dense training?

### Why Sparse Neural Networks?

<div class="container">
  <div class="row justify-content-center align-items-center">
      <div class="col-8 mt-3 mt-md-0">
        Sparse neural networks offer several compelling advantages:
        <ul>
        <li>For a fixed number of weights, they can offer better generalization and fewer Floating Point Operations (FLOPs) at inference.</li>
        <li>They hold the potential to significantly reduce the computational cost of training.</li>
        <li>They provide a way to learn the inherent structure of neural networks, moving beyond a "one-size-fits-all" dense architecture.</li>
        </ul>
      </div>
      <div class="col-4 mt-3 mt-md-0">
          {% include figure.liquid loading="eager" path="assets/img/srigl_sparsenn_simple.svg" title="Sparse Neural Network." class="img-fluid rounded z-depth-0" %}
      </div>
  </div>
</div>

## Paths to Sparsity: Pruning, Lottery Tickets, and Dynamic Sparse Training

Several approaches have been developed to tackle the challenge of obtaining sparse networks, these include:

### Traditional Pruning: Effective but Costly

<div class="container">
  <div class="row align-items-center justify-content-center">
      <div class="col-8 mt-3 mt-md-0 bg-white">
          {% include figure.liquid loading="eager" path="assets/img/srigl_sparsetraining_pruning.svg"  title="Pruning is highly effective, but requires dense pre-training." class="img-fluid rounded z-depth-0" %}
      </div>
  </div>
  <div class="caption">Figure: Pruning Dense Trained Models: Pruning a trained model is highly effective, but requires pre-trained models.</div>
</div>

Standard pruning techniques involve training a full, dense network and then removing (pruning) weights deemed less important, typically those with the smallest magnitude. This can be done once ("one-shot pruning") or iteratively. While effective at finding highly sparse subnetworks that retain accuracy, this still necessitates the expensive initial dense training.

### The Sparse Training Problem: Training with a Fixed Sparse Mask

<div class="container">
  <div class="row align-items-center">
      <div class="col-lg mt-3 mt-md-0 bg-white">
          {% include figure.liquid loading="eager" path="assets/img/srigl_sparsetraining_problem.svg"  title="The sparse training problem." class="img-fluid rounded z-depth-0" %}
      </div>
  </div>
  <div class="caption">Figure: Sparse Training Problem: Training a sparse mask from random initialization doesn't recover the generalization of a pruned dense model, even for a known good mask.</div>
</div>

Despite the success of pruning, simply training a sparse network from a random initialization, even with a known "good" sparse mask (the pattern of zeroed-out weights), often leads to poor performance compared to its dense counterpart or a pruned dense model. This is known as the sparse training problem.

The Lottery Ticket Hypothesis (LTH) <d-cite key="Frankle2019LTH"></d-cite> proposed that within a large, randomly initialized dense network, there exist smaller subnetworks (the "winning tickets") that, when trained in isolation from their original initialization weights (or weights from very early in training <d-cite key="Frankle2020LinearMode"></d-cite>), can achieve accuracy comparable to the full dense network. This was a foundational idea, suggesting that specific initializations within a sparse structure are crucial. However, finding these "winning tickets" is computationally expensive, especially for larger models, and the original findings were primarily on smaller datasets <d-cite key="Liu2019RethinkingPruning"></d-cite>. Research further showed that these Lottery Tickets often end up re-learning the solution that would have been found by pruning the dense model they originated from <d-cite key="Evci2022GradientFlow"></d-cite>.

### Dynamic Sparse Training (DST): Training Sparse from the Start

<div class="container">
  <div class="row align-items-center">
      <div class="col-lg mt-3 mt-md-0 bg-white">
          {% include figure.liquid loading="eager" path="assets/img/srigl_sparsetraining_dst.svg"  title="Dynamic Sparse Training (DST) methods are an alternative sparse training method that work well." class="img-fluid rounded z-depth-0" %}
      </div>
  </div>
  <div class="caption">Figure: Dynamic Sparse Training (DST) methods are an alternative sparse training, where the mask is changed over the course of training. DST methods can match the generalization performance of standard dense training.</div>
</div>

Dynamic Sparse Training (DST) methods offer a more direct approach to training sparse networks. Techniques like Sparse Evolutionary Training (SET) <d-cite key="Mocanu2018SET"></d-cite> and Rigging the Lottery Ticket (RigL) <d-cite key="Evci2020RigL"></d-cite> train networks that are sparse from initialization to the final solution ("sparse-to-sparse"). They achieve this by dynamically changing the sparse connectivity during training: periodically pruning less salient connections (e.g., small magnitude weights) and growing new ones (e.g., where the gradient magnitude is large). DST can achieve generalization comparable to dense training at high sparsity levels.

<!-- [Placeholder for Figure 3: Diagram illustrating the Dynamic Sparse Training process with grow/prune mask updates. (Based on PPT Slide 18)] -->

## Unstructured vs. Structured Sparsity

<div class="container">
  <div class="row align-items-center">
      <div class="col-lg mt-3 mt-md-0 bg-white">
          {% include figure.liquid loading="eager" path="assets/img/srigl_unstructuredvsstructured.svg"  title="Unstructured vs. Structured Sparsity." class="img-fluid rounded z-depth-0" %}
      </div>
  </div>
  <div class="caption">Figure: Unstructured vs. Structured sparsity. Unstructured sparsity is difficult to use efficiently on current computational hardware, such as CPUs and GPUs.</div>
</div>

### Unstructured Sparsity

A significant challenge with many DST methods like RigL is that they typically produce **unstructured sparsity**. This means individual weights are zeroed out irregularly across the weight matrices.

<div class="container">
  <div class="row align-items-center justify-content-center">
      <div class="col-8 mt-3 mt-md-0 bg-white">
          {% include figure.liquid loading="eager" path="assets/img/srigl_sparsetraining_unstructuredwmatrix.svg"  title="Untructured Sparsity." class="img-fluid rounded z-depth-0" %}
      </div>
  </div>
  <div class="caption">Figure: Unstructured Sparsity. A neural network with unstructured sparse weights, and the corresponding weight matrix.</div>
</div>

- **Pros:** Can achieve excellent generalization at very high sparsities (85-95%); fewer theoretical FLOPs.
- **Cons:** Poorly supported by standard hardware (CPUs/GPUs) and acceleration libraries, meaning theoretical speedups often don't translate into real-world gains.

### Structured Sparsity (e.g., removing neurons/blocks)

<div class="container">
  <div class="row align-items-center justify-content-center">
      <div class="col-8 mt-3 mt-md-0 bg-white">
          {% include figure.liquid loading="eager" path="assets/img/srigl_sparsetraining_structuredwmatrix.svg"  title="Structured Sparsity." class="img-fluid rounded z-depth-0" %}
      </div>
  </div>
  <div class="caption">Figure: Structured Sparsity. A neural network with structured sparse weights, and the corresponding weight matrix.</div>
</div>

In contrast, **structured sparsity** involves removing entire blocks of weights, such as channels, filters, or even neurons.

- **Pros:** Much better hardware support, leading to practical speedups as it often results in effectively smaller dense operations.
- **Cons:** Often leads to poorer generalization compared to unstructured sparsity at the same overall sparsity level, as it's a coarser form of pruning.

### N:M Fine-grained Structured Sparsity

<div class="container">
  <div class="row align-items-center justify-content-center">
      <div class="col-8 mt-3 mt-md-0 bg-white">
          {% include figure.liquid loading="eager" path="assets/img/srigl_sparsetraining_nmwmatrix.svg"  title="N:M Structured Sparsity." class="img-fluid rounded z-depth-0" %}
      </div>
  </div>
  <div class="caption">Figure: N:M Structured Sparsity. A neural network with N:M structured sparse weights, and the corresponding weight matrix.</div>
</div>

N:M fine-grained sparsity is a compromise where, within small contiguous blocks of M weights, exactly N weights are non-zero. NVIDIA's Ampere GPUs support 2:4 sparsity, offering some acceleration <d-cite key="Mishra2021Accelerating,Nvidia2020Ampere"></d-cite>.

The ideal scenario is to combine the high accuracy of unstructured DST with the hardware-friendliness of fine-grained structured sparsity.

## Dynamic Sparse Training for Learning Structured Sparse Representations

Our work "Dynamic Sparse Training with Structured Sparsity" <d-cite key="LasbyLasby2024SRigL"></d-cite> proposes a novel DST method called Structured RigL (SRigL) to address this challenge. SRigL aims to learn sparse networks that are both highly accurate _and_ possess a structure amenable to real-world acceleration.

### Constant Fan-in Sparsity

SRigL modifies the RigL algorithm to enforce a **constant fan-in** constraint. This means each neuron (or output channel in a convolutional layer) has the same number of active incoming connections. This is a specific type of N:M sparsity (where N is the fan-in and M is the potential dense fan-in) and results in a regular structure within the weight matrices. Theoretical analysis suggests that this constant fan-in constraint should not inherently impair training dynamics and might even offer slightly better output-norm variance compared to less constrained sparsity patterns, especially for very sparse networks.

### The Hidden Trick of DST: Neuron Ablation

<div class="container">
  <div class="row align-items-center justify-content-center">
      <div class="col-lg mt-3 mt-md-0 bg-white">
          {% include figure.liquid loading="eager" path="assets/img/srigl_sparsetraining_neuronablation.svg"  title="Neuron Ablation." class="img-fluid rounded z-depth-0" %}
      </div>
  </div>
  <div class="caption">Figure: Neuron Ablation in DST. When a neuron has too few weight to learn useful representations, DST methods learn to ablate, or remove, the entire neuron &mdash; effectively reducing the width of the layer.</div>
</div>

A key empirical finding was that standard unstructured DST methods, like RigL, when pushed to very high sparsity levels (>90%), implicitly learn to **ablate neurons** &mdash; that is, it they learn to remove neurons with very few weights, effectively reducing the width of layers.

This neuron ablation appears crucial for maintaining generalization at extreme sparsities, but enforcing a naive constant fan-in constraint would prevent this, as it would force every neuron to maintain the same number of weights, even if those weights are not useful for learning.

### The SRigL Algorithm

<div class="container">
  <div class="row align-items-center justify-content-center">
      <div class="col-lg mt-3 mt-md-0 bg-white">
          {% include figure.liquid loading="eager" path="assets/img/srigl_constantfanin.svg"  title="Constant Fan-in Fine-grained Sparsity." class="img-fluid rounded z-depth-0" %}
      </div>
  </div>
  <div class="caption">Figure: Constant Fan-in Fine-grained Sparsity. A form of N:M sparsity where every neuron on a layer is constained to have the same number of weights.</div>
</div>

SRigL integrates the constant fan-in objective with an explicit neuron ablation mechanism. The core steps, adapted from RigL, are:

1.  Identify weights to prune (smallest magnitude) and potential connections to grow (largest gradient magnitude on zeroed weights).
2.  Count salient weights per neuron.
3.  **Ablate neurons**: If a neuron has fewer salient weights than a defined threshold ($\gamma_{sal}$ multiplied by the target fan-in), it's entirely pruned. Its designated weights are redistributed.
4.  Compute the new constant fan-in based on any ablated neurons.
5.  Prune the globally smallest magnitude weights.
6.  For each _active_ neuron, regrow connections to meet the target constant fan-in, prioritizing those with the largest gradient magnitudes.

This allows SRigL to learn both fine-grained constant fan-in sparsity _within_ active neurons and coarser neuron-level structured sparsity.

<!--
**Algorithm 1: "Condensed" linear layer with constant fan-in sparsity forward pass**

1.  **Input:**
    - `x`: the input matrix of shape (`batch_size`, `num_features`)
    - `w`: the condensed weight matrix of shape (`active_neurons`, `constant_fan_in`)
    - `indx`: indices of non-zero dense weights of shape (`active_neurons`, `constant_fan_in`)
2.  `output` <- `torch.zeros(size=(batch_size, neurons))`
3.  **For** `b in range(batch_size)`: // For each sample in mini-batch
4.  &nbsp;&nbsp;&nbsp;&nbsp;**For** `n in range(neurons)`: // For each active neuron in layer
5.  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**For** `k in range(constant_fan_in)`: // For each non-zero weight
6.  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`source_idx` <- `idx[n, k]`
7.  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`feature` <- `x[b, source_idx]`
8.  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`output[b, n]` += `feature * w[n, k]`
9.  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**EndFor**
10. &nbsp;&nbsp;&nbsp;&nbsp;**EndFor**
11. **EndFor**
12. **Return** `output` -->

## Key Results: Performance and Acceleration

SRigL was evaluated on image classification tasks using CIFAR-10 (ResNet-18, Wide ResNet-22) and ImageNet (ResNet-50, MobileNet-V3, ViT-B/16) <d-cite key="Lasby2024SRigL"></d-cite>.

### Matching Dense Accuracy with Structured Sparsity

<div class="container">
  <div class="row align-items-center justify-content-center">
      <div class="col-md-6 mt-3 mt-md-0 bg-white">
          {% include figure.liquid loading="eager" path="assets/img/srigl_resnet18.svg"  title="ResNet-18/CIFAR-10" class="img-fluid rounded z-depth-0" %}
      </div>
      <div class="col-md-6 mt-3 mt-md-0 bg-white">
          {% include figure.liquid loading="eager" path="assets/img/srigl_wide_resnet22.svg"  title="Wide ResNet 22/CIFAR-10" class="img-fluid rounded z-depth-0" %}
      </div>
      <div class="col-md-6 mt-3 mt-md-0 bg-white">
          {% include figure.liquid loading="eager" path="assets/img/srigl_resnet50.svg"  title="ResNet-50/ImageNet" class="img-fluid rounded z-depth-0" %}
      </div>
      <div class="col-md-6 mt-3 mt-md-0 bg-white">
          {% include figure.liquid loading="eager" path="assets/img/srigl_vit.svg"  title="ViT/ImageNet" class="img-fluid rounded z-depth-0" %}
      </div>
  </div>
  <div class="caption">Figure: (a) Percentage active neurons (i.e., not ablated) following RigL/SRigL training on ResNet-50/ImageNet (b) Sparse Fan-In vs. ViT layer index at the end of training with RigL at 90% sparsity.</div>
</div>

SRigL with neuron ablation was shown to achieve generalization performance comparable to unstructured RigL and often close to the dense training baseline, even at high sparsities (e.g., 90-95%) across various architectures. Extended training further improved performance, similar to RigL.

### The Importance of Neuron Ablation

<div class="container">
  <div class="row align-items-center justify-content-center">
      <div class="col-md-6 mt-3 mt-md-0 bg-white">
          {% include figure.liquid loading="eager" path="assets/img/srigl_imagenet_perc_active.svg"  title="ResNet-50/ImageNet." class="img-fluid rounded z-depth-0" %}
      </div>
      <div class="col-md-6 mt-3 mt-md-0 bg-white">
          {% include figure.liquid loading="eager" path="assets/img/srigl_vit_rigl_fan_in.svg"  title="Sparse Fan-in vs VIT layer index." class="img-fluid rounded z-depth-0" %}
      </div>
  </div>
  <div class="caption">Figure: (a) Percentage active neurons (i.e., not ablated) following RigL/SRigL training on ResNet-50/ImageNet (b) Sparse Fan-in vs. ViT layer index at the end of training with RigL at 90% sparsity.</div>
</div>

The neuron ablation component was critical. Without it, SRigL's performance lagged behind unstructured RigL at very high sparsities (>90%) and with Vision Transformers. Enabling SRigL to ablate neurons restored performance to RigL levels. The percentage of active neurons (not ablated) learned by SRigL dynamically adapted with sparsity, mirroring RigL's behavior. For Vision Transformers, SRigL's performance was particularly sensitive to the ablation threshold $\gamma_{sal}$, with higher thresholds performing best, suggesting that aggressively ablating neurons to maintain sufficient density in the remaining ones is beneficial for ViTs.

### Real-World Speedups

<div class="container">
  <div class="row align-items-center justify-content-center">
      <div class="col-md-12 mt-3 mt-md-0 bg-white">
          {% include figure.liquid loading="eager" path="assets/img/srigl_grouped-bar-threads-4.svg"  title="ResNet-50/ImageNet." class="img-fluid rounded z-depth-0" %}
      </div>
      <div class="col-md-12 mt-3 mt-md-0 bg-white">
          {% include figure.liquid loading="eager" path="assets/img/srigl_grouped-bar-threads-4-device-gpu-batch_size-2048-broken-y.svg"  title="Sparse Fan-in vs VIT layer index." class="img-fluid rounded z-depth-0" %}
      </div>
  </div>
  <div class="caption">Figure: (a) Percentage active neurons (i.e., not ablated) following RigL/SRigL training on ResNet-50/ImageNet (b) Batched GPU inference with batch size of 2048 on an NVIDIA Titan V. At 90% sparsity, our condensed representation is 1.7$\times$ faster than dense and 13.0$\times$ faster than unstructured (CSR) sparse layers. Note y-axis is log-scaled..</div>
</div>

The structured sparsity learned by SRigL (constant fan-in + ablated neurons) translates into tangible inference speedups. The paper demonstrates a "condensed" matrix multiplication method (Algorithm 1 in the paper ) that leverages this structure.

- **CPU (Online Inference, single input):** At 90% sparsity, SRigL's condensed representation was up to **3.4x faster than dense** and 2.5x faster than unstructured (CSR) sparse layers on an Intel Xeon CPU.
- **GPU (Batched Inference, batch size 256):** At 90% sparsity, it was **1.7x faster than dense** and 13.0x faster than unstructured (CSR) sparse layers on an NVIDIA Titan V GPU.

These speedups are achieved even with a straightforward PyTorch implementation, highlighting the practical benefits of the learned structure.

## Conclusion and Future Horizons

"Dynamic Sparse Training with Structured Sparsity" <d-cite key="Lasby2024SRigL"></d-cite> makes a significant stride towards practical sparse neural networks. SRigL demonstrates that it's possible to:

- Train networks from a sparse initialization to a sparse solution (sparse-to-sparse).
- Achieve generalization performance on par with state-of-the-art _unstructured_ sparse training methods.
- Learn a combination of fine-grained constant fan-in and neuron-level structured sparsity.
- Realize significant real-world inference acceleration on both CPUs and GPUs due to this learned structure.

The insight that successful DST methods at high sparsity inherently learn to reduce model width (neuron ablation) is key and SRigL formalizes this. This work underscores that much of the progress in deep learning comes from methods that better leverage hardware capabilities.

Future directions include:

- Improving the convergence speed of DST methods, which can take longer to train than dense models.
- Exploring the potential of DST to learn novel, efficient architectures for new data domains beyond typical NLP/CV tasks, particularly in areas like "AI for Science."

SRigL paves the way for deploying highly efficient and accurate sparse models in a wider range of applications, making powerful AI more accessible and sustainable.
