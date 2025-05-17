---
layout: distill
title: "Dynamic Sparse Training with Structured Sparsity: Sparse Training for Real-World Hardware Acceleration"
description: "Exploring how Dynamic Sparse Training can learn neural networks that are not only highly sparse and accurate but also structured for real-world hardware acceleration, based on the SRigL method."
tags: dynamic sparse training structured sparsity compression RigL SET
giscus_comments: false
date: 2024-11-14 # Date of the presentation
featured: false
authors:
  - name: Mike Lasby
    affiliations:
      name: University of Calgary
  - name: Anna Golubeva
    affiliations:
      name: MIT, IAIFI
  - name: Utku Evci
    affiliations:
      name: Google DeepMind
  - name: Mihai Nica
    affiliations:
      name: University of Guelph, Vector Institute for AI
  - name: Yani Ioannou
    url: "mailto:yani.ioannou@ucalgary.ca" # From PPT
    affiliations:
      name: University of Calgary
bibliography: 2025-05-17-training-sparse-structured-nn.bib

doi: 10.48550/arXiv.2305.02299
paper_url: https://openreview.net/pdf?id=kOBkxFRKTA
# _styles: >
#   .fake-img {
#     background: #bbb;
#     border: 1px solid rgba(0, 0, 0, 0.1);
#     box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
#     margin-bottom: 12px;
#   }
#   .fake-img p {
#     font-family: monospace;
#     color: white;
#     text-align: left;
#     margin: 12px 0;
#     text-align: center;
#     font-size: 16px;
#   }
---

## TL;DR

The challenge in training sparse neural networks is to achieve both high accuracy and practical hardware acceleration. Unstructured sparsity often yields good performance but is hard to speed up, while traditional structured sparsity can hurt performance.

Our International Conference in Learning Representations (ICLR) 2024 paper, ["Dynamic Sparse Training with Structured Sparsity"](https://openreview.net/forum?id=kOBkxFRKTA) <d-cite key="LasbyLasby2024SRigL"></d-cite> introduces Structured RigL (SRigL) to addresses this by dynamically learning hardware-friendly sparse weight representations without sacrificing accuracy.

Key findings of the work include:

- SRigL successfully learns a combination of fine-grained N:M structured sparsity (constant fan-in) and neuron-level sparsity (neuron ablation) dynamically from a sparse initialization.
- The explicit integration of neuron ablation, a behavior implicitly learned by unstructured DST methods at high sparsities, is crucial for SRigL to match the generalization performance of dense and unstructured sparse models, even at extreme sparsities (up to 99%).
- The learned structured sparsity enables a "condensed sparse representation," which translates to significant real-world inference speedups on commodity CPUs (up to $3.4 \times$ vs. dense at 90% sparsity) and GPUs (up to $1.7 \times$ vs. dense at 90% sparsity), outperforming unstructured sparse formats in many practical scenarios.
- SRigL demonstrates a viable path to train sparse neural networks that are both highly accurate and practically efficient, bridging the gap between unstructured DST performance and structured sparsity acceleration.

## The Quest for Efficient Neural Networks

State-of-the-art deep neural networks (DNNs) have achieved remarkable feats, but their ever-increasing size brings ballooning training costs, often outstripping Moore's Law[cite: 58]. This trend makes cutting-edge AI research less accessible. While techniques exist to prune trained dense models, effectively reducing their parameter count by 85-95% without sacrificing generalization[cite: 15, 49], the initial dense training phase remains a significant burden. Can we train these sparse, efficient networks from scratch?

### Why Sparse Neural Networks?

Sparse neural networks offer several compelling advantages[cite: 61]:

- For a fixed number of weights, they can offer better generalization and fewer Floating Point Operations (FLOPs) at inference.
- They hold the potential to significantly reduce the computational cost of training.
- They provide a way to learn the inherent structure of neural networks, moving beyond a "one-size-fits-all" dense architecture.

### The Challenge: The Sparse Training Problem

Despite the success of pruning, simply training a sparse network from a random initialization, even with a known "good" sparse mask (the pattern of zeroed-out weights), often leads to poor performance compared to its dense counterpart or a pruned dense model[cite: 16, 60]. This is known as the sparse training problem.

[Placeholder for Figure 1: Diagram illustrating the difference in outcome between pruning a trained dense model vs. attempting to train a sparse model from scratch with a fixed mask. (Based on PPT Slide 16, 60)]

## Paths to Sparsity: Pruning, Lottery Tickets, and Dynamic Sparse Training

Several approaches have been developed to tackle the challenge of obtaining efficient sparse networks.

### Traditional Pruning: Effective but Costly

Standard pruning techniques involve training a full, dense network and then removing (pruning) weights deemed less important, typically those with the smallest magnitude[cite: 15, 57]. This can be done once ("one-shot pruning") or iteratively. While effective at finding highly sparse subnetworks that retain accuracy, this still necessitates the expensive initial dense training.

### The Lottery Ticket Hypothesis: A Glimpse of Hope

The Lottery Ticket Hypothesis (LTH) <d-cite key="Frankle2019LTH"></d-cite> proposed that within a large, randomly initialized dense network, there exist smaller subnetworks (the "winning tickets") that, when trained in isolation from their original initialization weights (or weights from very early in training <d-cite key="Frankle2020LinearMode"></d-cite>), can achieve accuracy comparable to the full dense network. This was a foundational idea, suggesting that specific initializations within a sparse structure are crucial. However, finding these "winning tickets" is computationally expensive, especially for larger models, and the original findings were primarily on smaller datasets <d-cite key="Liu2019RethinkingPruning"></d-cite>. Research further showed that these Lottery Tickets often end up re-learning the solution that would have been found by pruning the dense model they originated from <d-cite key="Evci2022GradientFlow"></d-cite>.

[Placeholder for Figure 2: Conceptual diagram of the Lottery Ticket Hypothesis, showing a dense network, a winning ticket subnetwork, and its training from original initial weights. (Based on PPT Slides 68, 76, 94)]

### Dynamic Sparse Training (DST): Training Sparse from the Start

Dynamic Sparse Training (DST) methods offer a more direct approach to training sparse networks. Techniques like Sparse Evolutionary Training (SET) <d-cite key="Mocanu2018SET"></d-cite> and Rigging the Lottery Ticket (RigL) <d-cite key="Evci2020RigL"></d-cite> train networks that are sparse from initialization to the final solution ("sparse-to-sparse"). They achieve this by dynamically changing the sparse connectivity during training: periodically pruning less salient connections (e.g., small magnitude weights) and growing new ones (e.g., where the gradient magnitude is large)[cite: 17, 18]. DST can achieve generalization comparable to dense training at high sparsity levels.

[Placeholder for Figure 3: Diagram illustrating the Dynamic Sparse Training process with grow/prune mask updates. (Based on PPT Slide 18)]

## The Bottleneck: Unstructured vs. Structured Sparsity

A significant challenge with many DST methods like RigL is that they typically produce **unstructured sparsity**[cite: 21, 143]. This means individual weights are zeroed out irregularly across the weight matrices.

- **Unstructured Sparsity:**
  - **Pros:** Can achieve excellent generalization at very high sparsities (85-95%); fewer theoretical FLOPs.
  - **Cons:** Poorly supported by standard hardware (CPUs/GPUs) and acceleration libraries, meaning theoretical speedups often don't translate into real-world gains[cite: 25, 138].

In contrast, **structured sparsity** involves removing entire blocks of weights, such as channels, filters, or even neurons.

- **Structured Sparsity (e.g., removing neurons/blocks):**
  - **Pros:** Much better hardware support, leading to practical speedups as it often results in effectively smaller dense operations.
  - **Cons:** Often leads to poorer generalization compared to unstructured sparsity at the same overall sparsity level, as it's a coarser form of pruning.
- **N:M Fine-grained Structured Sparsity:** A compromise where, within small contiguous blocks of M weights, exactly N weights are non-zero. NVIDIA's Ampere GPUs support 2:4 sparsity, offering some acceleration <d-cite key="Mishra2021Accelerating, Nvidia2020Ampere"></d-cite>.

The ideal scenario is to combine the high accuracy of unstructured DST with the hardware-friendliness of structured sparsity.

[Placeholder for Figure 4: Visual comparison of unstructured, block-structured, and N:M structured sparsity patterns. (Based on PPT Slides 25, 26)]

## Introducing Structured RigL (SRigL): DST Meets Structure

The work "Dynamic Sparse Training with Structured Sparsity" <d-cite key="Lasby2024SRigL"></d-cite> (which this post primarily discusses, based on the presentation [cite: 1, 12, 29, 46] and paper [cite: 137]) proposes a novel DST method called Structured RigL (SRigL) to address this challenge. SRigL aims to learn sparse networks that are both highly accurate _and_ possess a structure amenable to real-world acceleration.

### Constant Fan-in Sparsity

SRigL modifies the RigL algorithm to enforce a **constant fan-in** constraint. This means each neuron (or output channel in a convolutional layer) has the same number of active incoming connections[cite: 139, 159, 237]. This is a specific type of N:M sparsity (where N is the fan-in and M is the potential dense fan-in) and results in a regular structure within the weight matrices[cite: 160]. Theoretical analysis suggests that this constant fan-in constraint should not inherently impair training dynamics and might even offer slightly better output-norm variance compared to less constrained sparsity patterns, especially for very sparse networks.

### The Hidden Trick of DST: Neuron Ablation

A key empirical finding was that standard unstructured RigL, when pushed to very high sparsity levels (>90%), implicitly learns to **ablate neurons** – that is, it disconnects all incoming and outgoing connections to certain neurons, effectively reducing the width of layers. This neuron ablation appears crucial for maintaining generalization at extreme sparsities.

[Placeholder for Figure 5: Diagram illustrating neuron ablation where a neuron in layer L+1 becomes completely disconnected from layer L. (Based on PPT Slide 33 and Paper Figure 2 [cite: 239])]

A naive constant fan-in constraint would prevent this, as it would force every neuron to maintain its connections.

### The SRigL Algorithm

SRigL integrates the constant fan-in objective with an explicit neuron ablation mechanism[cite: 140, 246]. The core steps, adapted from RigL, are:

1.  Identify weights to prune (smallest magnitude) and potential connections to grow (largest gradient magnitude on zeroed weights).
2.  Count salient weights per neuron.
3.  **Ablate neurons**: If a neuron has fewer salient weights than a defined threshold ($\gamma_{sal}$ multiplied by the target fan-in), it's entirely pruned. Its designated weights are redistributed.
4.  Compute the new constant fan-in based on any ablated neurons.
5.  Prune the globally smallest magnitude weights.
6.  For each _active_ neuron, regrow connections to meet the target constant fan-in, prioritizing those with the largest gradient magnitudes.

This allows SRigL to learn both fine-grained constant fan-in sparsity _within_ active neurons and coarser neuron-level structured sparsity.

## Key Results: Performance and Acceleration

SRigL was evaluated on image classification tasks using CIFAR-10 (ResNet-18, Wide ResNet-22) and ImageNet (ResNet-50, MobileNet-V3, ViT-B/16) <d-cite key="Lasby2024SRigL"></d-cite>.

### Matching Dense Accuracy with Structured Sparsity

SRigL with neuron ablation was shown to achieve generalization performance comparable to unstructured RigL and often close to the dense training baseline, even at high sparsities (e.g., 90-95%) across various architectures. Extended training further improved performance, similar to RigL[cite: 268, 278].

[Placeholder for Figure 6: Graphs showing Test Accuracy vs. Sparsity for SRigL (with and without ablation) compared to RigL and a dense benchmark on ImageNet/ResNet-50 and ViT-B/16. (Based on Paper Figure 3a[cite: 267], Table 4 [cite: 305] and PPT Slides 31, 35, 37)]

### The Importance of Neuron Ablation

The neuron ablation component was critical. Without it, SRigL's performance lagged behind unstructured RigL at very high sparsities (>90%) and with Vision Transformers. Enabling SRigL to ablate neurons restored performance to RigL levels[cite: 153, 242, 279, 319]. The percentage of active neurons (not ablated) learned by SRigL dynamically adapted with sparsity, mirroring RigL's behavior[cite: 269, 270]. For Vision Transformers, SRigL's performance was particularly sensitive to the ablation threshold $\gamma_{sal}$, with higher thresholds performing best, suggesting that aggressively ablating neurons to maintain sufficient density in the remaining ones is beneficial for ViTs.

[Placeholder for Figure 7: Graph showing Percentage of Active Neurons vs. Sparsity for SRigL and RigL. (Based on Paper Figure 3b [cite: 269] and PPT Slides 32, 34)]

### Real-World Speedups

The structured sparsity learned by SRigL (constant fan-in + ablated neurons) translates into tangible inference speedups. The paper demonstrates a "condensed" matrix multiplication method (Algorithm 1 in the paper [cite: 321]) that leverages this structure.

- **CPU (Online Inference, single input):** At 90% sparsity, SRigL's condensed representation was up to **3.4x faster than dense** and 2.5x faster than unstructured (CSR) sparse layers on an Intel Xeon CPU[cite: 141, 338, 339].
- **GPU (Batched Inference, batch size 256):** At 90% sparsity, it was **1.7x faster than dense** and 13.0x faster than unstructured (CSR) sparse layers on an NVIDIA Titan V GPU[cite: 141, 340].

These speedups are achieved even with a straightforward PyTorch implementation, highlighting the practical benefits of the learned structure[cite: 324, 330, 335].

[Placeholder for Figure 8: Bar charts comparing inference timings (CPU and GPU) for SRigL (condensed), structured-only, unstructured CSR, and dense layers at various sparsities. (Based on Paper Figure 4 [cite: 335] and PPT Slide 40 (bottom charts))]

## Under the Hood: Gradient Flow and Initialization

Earlier work <d-cite key="Evci2022GradientFlow"></d-cite> (also from the Calgary ML Lab, [cite: 1, 4, 5, 83, 88, 125, 133]) investigated _why_ training sparse networks from random initializations is so challenging. Key findings included:

- Sparse networks often suffer from poor **gradient flow** at initialization and during early training compared to dense networks.
- Sparsity-aware initializations that account for the actual fan-in of each neuron can improve gradient flow at initialization.
- Dynamic Sparse Training methods like RigL appear to mitigate some of these gradient flow issues during training.
- The success of Lottery Ticket initializations (especially with weight rewinding) seems less about fundamentally better gradient flow and more about "nudging" the optimization towards re-learning the (good) pruned solution from which the ticket was derived[cite: 90, 91, 92, 125].

SRigL builds upon these understandings by using a robust DST backbone (RigL) which inherently handles some of these dynamic optimization challenges, while focusing on imparting a hardware-friendly structure.

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
