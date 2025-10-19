---
layout: distill
title: "Winning the Other Lottery: Aligning Masks for Sparse Training"
description: "An exploration of why Lottery Ticket masks fail on new random initializations and how understanding weight symmetry allows us to successfully reuse them."
date: 2025-07-14
last_updated: 2025-07-14
post_author: Yani Ioannou
authors:
  - name: Mohammed Adnan
    url: "https://www.linkedin.com/in/adnan-mohammed-/" # Placeholder URL
    affiliations:
      name: University of Calgary
  - name: Rohan Jain
    url: "https://www.linkedin.com/in/rohan-jain-6a1b2c3d/" # Placeholder URL
    affiliations:
      name: University of Calgary
  - name: Ekansh Sharma
    url: "https://www.linkedin.com/in/ekansh-sharma-/" # Placeholder URL
    affiliations:
      name: University of Toronto
  - name: Rahul G. Krishnan
    url: "https://www.cs.toronto.edu/~rahulgk/" # Placeholder URL
    affiliations:
      name: University of Toronto
  - name: Yani Ioannou
    url: "https://yani.ai/"
    affiliations:
      name: University of Calgary
paper_url: https://arxiv.org/abs/2505.05143
doi: # To be added upon publication
bibliography: 2025-07-13-sparse-rebasin.bib
thumbnail: assets/img/sparse-rebasin/sparsebasin_sparsetrainingproblem.svg
pretty_table: true

toc: true
related_posts: true
---

## TL;DR

The Lottery Ticket Hypothesis (LTH) provides a recipe for finding remarkably sparse "winning ticket" networks that can be trained to match the performance of their dense counterparts. However, there's a catch: a winning ticket's sparse mask is tightly coupled to the _original weight initialization_ used to find it <d-cite key="frankle2019lth"></d-cite>. Using the same mask with a new random initialization (the "sparse training problem") results in a significant drop in performance, making LTH a costly procedure. Our **ICML 2025** paper "Sparse Training from Random Initialization: Aligning Lottery Ticket Masks using Weight Symmetry" <d-cite key="adnan2025sparse"></d-cite> investigates this from a weight-space symmetry perspective and finds:

- **The Problem is Misalignment:** The reason LTH masks don't generalize to new initializations is a misalignment of optimization basins in the loss landscape, which arises from the inherent permutation symmetries of neural networks. A mask found in one basin won't work well if the new initialization starts in a different, symmetrically equivalent basin <d-cite key="adnan2025sparse"></d-cite>.
- **The Solution is Alignment:** We can fix this! By finding the permutation that aligns the basins of two different models and applying that same permutation to the LTH mask, we can successfully train a sparse network from a _new_ random initialization <d-cite key="adnan2025sparse"></d-cite>.
- **Bridging the Performance Gap:** Training with this **permuted mask** significantly improves generalization compared to naively using the original mask, nearly matching the performance of the original LTH solution across various models (VGG, ResNet) and datasets (CIFAR-10/100, ImageNet) <d-cite key="adnan2025sparse"></d-cite>.
- **Unlocking Diversity:** Unlike standard LTH, which consistently relearns the same solution <d-cite key="evci2022gradientflow"></d-cite>, our permutation-based method can train more diverse solutions when starting from different random initializations. This leads to better-performing ensembles <d-cite key="adnan2025sparse"></d-cite>.

---

## The Lottery Ticket Puzzle

The quest for smaller, faster, and more efficient neural networks has led to exciting breakthroughs in network **sparsity**. One of the most influential ideas in this area is the **Lottery Ticket Hypothesis (LTH)** <d-cite key="frankle2019lth"></d-cite>. LTH suggests that within a large, dense network, there are sparse subnetworks (the "winning tickets") that are exceptionally good at training. The standard recipe is:

1.  Train a full, dense network.
2.  Prune the connections with the smallest magnitude weights to get a sparse mask.
3.  "Rewind" the weights of the remaining connections to their values from very early in training and train the sparse network again.

This process can produce sparse models that match the performance of the original dense one <d-cite key="frankle2019lth"></d-cite>. But what if we want to skip the expensive dense pre-training and just use a winning ticket mask to train a sparse model from a _new_ random start? This is the heart of the **sparse training problem**. Unfortunately, this doesn't work well; the performance drops dramatically <d-cite key="frankle2019lth,adnan2025sparse"></d-cite>. A winning ticket seems to be valid for one lottery drawing only. Why?

<div class="container">
  <div class="row justify-content-center align-items-center">
      <div class="col-lg mt-3 mt-md-0 bg-white">
          <img src="/assets/img/sparse-rebasin/sparsetrainingproblem2.svg" alt="Diagram showing the sparse training problem where a pruned mask applied to a new initialization performs poorly." class="img-fluid rounded z-depth-0" loading="eager" />
      </div>
  </div>
  <div class="caption">Figure 1: (Left) The standard pruning pipeline creates a good pruned solution. (Right) The sparse training problem: applying the mask from the pruned solution to a new, different random initialization results in poor performance.</div>
</div>

---

## It's All About Symmetry

The answer lies in a fundamental property of neural networks: **weight symmetry**. If you take a hidden layer in a network and swap two of its neurons—including their incoming and outgoing weights—the function the network computes remains identical <d-cite key="entezari2022role"></d-cite>. However, in the high-dimensional space of all possible weights, these two networks are at completely different locations.

<div class="container">
  <div class="row align-items-center justify-content-center">
      <div class="col-10 mt-3 mt-md-0">
          <img src="/assets/img/sparse-rebasin/weightsymmetry3.svg" alt="Diagram illustrating that swapping two neurons results in a functionally identical network." class="img-fluid rounded z-depth-0" loading="eager" />
          <div class="caption">Figure 2: Swapping neurons $h_1$ and $h_2$ (along with their corresponding weights) results in a different point in weight space, but the network's output is unchanged.</div>
      </div>
  </div>
</div>

This symmetry means the loss landscape is filled with many identical, mirrored "valleys" or **basins of attraction**. When we train a model from a random initialization, it descends into one of these basins. Recent work on Linear Mode Connectivity (LMC) suggests that most solutions found by SGD lie in a _single_ basin, once you account for these permutations <d-cite key="entezari2022role,ainsworth2023git"></d-cite>.

### The Hypothesis: A Tale of Two Basins

This brings us to our core hypothesis <d-cite key="adnan2025sparse"></d-cite>:
**An LTH mask fails on a new initialization because the mask is aligned to one basin, while the new random initialization has landed in another.**

The optimization process is essentially starting in the wrong valley for the map it's been given. Naively applying the mask pulls the new initialization far away from a good solution path, leading to poor performance <d-cite key="adnan2025sparse"></d-cite>.

<div class="container">
  <div class="row justify-content-center align-items-center">
      <div class="col-lg mt-3 mt-md-0 bg-white">
          <img src="/assets/img/sparse-rebasin/sparsebasin_sparsetrainingproblem.svg" alt="Loss landscape showing how a naive mask application fails, while a permuted mask succeeds." class="img-fluid rounded z-depth-0" loading="eager" />
      </div>
  </div>
  <div class="caption">Figure 3: An illustration of our hypothesis <d-cite key="adnan2025sparse"></d-cite>. (a) A dense model A is trained and pruned, defining a mask $m_A$. (b) Training model B from a new initialization with mask $m_A$ (red path) fails. Our solution is to permute the mask to $\pi(m_A)$, which aligns with model B's basin and enables successful sparse training (green path).</div>
</div>

But what if we could "rotate" the mask to match the orientation of the new basin? This is exactly what we propose.

---

## The Method: Aligning Masks with Permutations

Our method leverages recent advances in model merging, like Git Re-Basin <d-cite key="ainsworth2023git"></d-cite>, which find the permutation that aligns the neurons of two separately trained models. Our training paradigm is as follows <d-cite key="adnan2025sparse"></d-cite>:

1.  **Train Two Dense Models:** Start with two different random initializations, `w_A` and `w_B`, and train them to convergence to get two dense models, `Model A` and `Model B`.
2.  **Find the Permutation:** Use an **activation matching** algorithm <d-cite key="jordan2023repair"></d-cite> to find the permutation, `π`, that best aligns the neurons of `Model A` with `Model B`. This essentially finds the "rotation" needed to map one solution basin onto the other.
3.  **Get the LTH Mask:** Prune `Model A` using standard iterative magnitude pruning (IMP) to get a sparse "winning ticket" mask, `m_A`.
4.  **Permute the Mask:** Apply the permutation `π` to the mask to get a new, aligned mask: `π(m_A)`.
5.  **Train from Scratch (Almost)!** Train a new sparse model starting from the `w_B` initialization (rewound to an early checkpoint, `k`), but using the **permuted mask** `π(m_A)`.

<div class="container">
  <div class="row justify-content-center align-items-center">
      <div class="col-lg mt-3 mt-md-0 bg-white">
          <img src="assets/img/sparse-rebasin/method_overview.png" alt="Diagram of the training paradigm, from training dense models to permutation matching and final sparse training." class="img-fluid rounded z-depth-0" loading="eager" />
      </div>
  </div>
  <div class="caption">Figure 4: The overall framework of our training procedure <d-cite key="adnan2025sparse"></d-cite>. We use two trained dense models to find a permutation `π`. This permutation is then applied to the mask from Model A, allowing it to be successfully used to train Model B from a random initialization.</div>
</div>

---

## The Results: It Works!

Across a wide range of experiments, our method demonstrates that aligning the mask is the key to solving the sparse training problem for LTH masks <d-cite key="adnan2025sparse"></d-cite>.

### Closing the Performance Gap

When we compare the performance of the standard `LTH` solution, the `Naive` solution (un-permuted mask on a new init), and our `Permuted` solution, the results are clear. The `Permuted` approach consistently and significantly outperforms the `Naive` baseline, closing most of the performance gap to the original `LTH` solution. The effect is especially pronounced at higher sparsity levels, where the `Naive` method struggles most <d-cite key="adnan2025sparse"></d-cite>.

<div class="container">
  <div class="row justify-content-center align-items-center bg-white">
      <div class="col-10 mt-3 mt-md-0">
          <img src="assets/img/sparse-rebasin/results_resnet_cifar10.png" alt="Graphs showing test accuracy vs rewind points for ResNet20 on CIFAR-10 at different sparsity levels and widths." class="img-fluid rounded z-depth-0" loading="eager" />
      </div>
  </div>
  <div class="caption">Figure 5: Test accuracy on CIFAR-10 for ResNet20 of varying widths (`w`) and sparsities. The permuted solution (blue) consistently outperforms the naive one (orange) and gets closer to the LTH baseline (green), especially as model width increases <d-cite key="adnan2025sparse"></d-cite>.</div>
</div>

### Wider is Better, and More Diverse

We found that our method works even better on wider models. Wider networks have smoother loss landscapes, which allows the activation matching algorithm to find a more accurate permutation, reducing the loss barrier between basins <d-cite key="adnan2025sparse"></d-cite>.

<div class="container">
  <div class="row justify-content-center align-items-center">
      <div class="col-10 mt-3 mt-md-0 bg-white">
          <img src="assets/img/sparse-rebasin/diversity_table.png" alt="Table showing ensemble diversity metrics for different sparse models." class="img-fluid rounded z-depth-0" loading="eager" />
      </div>
  </div>
  <div class="caption">Table 1: Functional diversity analysis on CIFAR-100 <d-cite key="adnan2025sparse"></d-cite>. An ensemble of `permuted` models is far more diverse (higher disagreement, KL, JS) than an `LTH` ensemble. This increased diversity leads to a much larger boost in ensemble accuracy, similar to ensembles of independently pruned (IMP) models.</div>
</div>

Furthermore, by starting from different random initializations, our method can produce a more **diverse set of solutions** compared to LTH, which is known to be functionally very similar to the pruned model it came from <d-cite key="evci2022gradientflow"></d-cite>. This diversity is beneficial: an ensemble of our `permuted` models achieves significantly higher accuracy than an ensemble of `LTH` models, which are too similar to provide much of a boost <d-cite key="adnan2025sparse"></d-cite>.

---

## Key Insights Summarized

This investigation into aligning sparse masks reveals:

1.  **Symmetry is the Culprit:** The failure of LTH masks to transfer to new initializations is not a fundamental flaw of sparsity, but a consequence of weight permutation symmetry and the resulting misalignment of optimization basins <d-cite key="adnan2025sparse"></d-cite>.
2.  **Permutation is the Solution:** By explicitly finding and correcting for this misalignment—by permuting the sparse mask—we can successfully reuse a winning ticket to train a high-performing sparse model from a completely new random initialization <d-cite key="adnan2025sparse"></d-cite>.
3.  **Diversity is a Feature:** This approach not only solves a practical problem with LTH but also opens the door to finding more functionally diverse sparse solutions than LTH alone, leading to more powerful ensembles <d-cite key="adnan2025sparse"></d-cite>.
4.  **Wider Models Align Better:** The effectiveness of the permutation alignment increases with model width, correlating with a reduction in the loss barrier between symmetrically equivalent solutions <d-cite key="adnan2025sparse"></d-cite>.

## Conclusion and Future Directions

This work provides a new lens through which to view the sparse training problem. The success of a lottery ticket isn't just about the mask's structure, but also its _alignment_ with the optimization landscape. While our method requires training two dense models to find the permutation and is thus a tool for insight rather than efficiency, it proves a crucial point: winning ticket masks are more portable than previously thought <d-cite key="adnan2025sparse"></d-cite>.

This opens up exciting new questions. Can we find these permutations more efficiently? Can we design initialization schemes that are "symmetry-aware" and land in a canonical basin by default? By showing that the barriers of sparse training can be overcome by understanding its underlying geometry, we hope to spur future work that makes training sparse models from scratch a practical and powerful reality <d-cite key="adnan2025sparse"></d-cite>.

---

## Citing our work

If you find this work useful, please consider citing it using the following BibTeX entry:

```bibtex
@inproceedings{adnan2025sparse,
  author = {Adnan, Mohammed and Jain, Rohan and Sharma, Ekansh and Krishnan, Rahul G. and Ioannou, Yani},
  title = {Sparse Training from Random Initialization: Aligning Lottery Ticket Masks using Weight Symmetry},
  year = {2025},
  booktitle = {Proceedings of the Forty-Second International Conference on Machine Learning (ICML)},
  venue = {Vienna, Austria},
}
```
