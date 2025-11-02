---
layout: distill
title: "Sparse Training from Random Initialization: Aligning Lottery Ticket Masks using Weight Symmetry"
description: "An exploration of why Lottery Ticket Hypothesis masks fail on new random initializations and how understanding weight symmetry in neural networks allows us to successfully reuse them."
date: 2025-07-14
last_updated: 2025-11-01
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
paper_url: https://proceedings.mlr.press/v267/adnan25a.html
doi: 10.48550/arXiv.2505.05143
bibliography: 2025-07-13-sparse-rebasin.bib
thumbnail: assets/img/sparse-rebasin/sparsebasin_sparsetrainingproblem.svg
pretty_table: true

toc: true
related_posts: true
---

## TL;DR

The Lottery Ticket Hypothesis (LTH) demonstrates that remarkably sparse "winning ticket" neural network models can be trained to match the performance of their dense counterparts. However, there's a catch: a winning ticket's sparse mask is tightly coupled to the _original weight initialization_ used to find it <d-cite key="frankle2019lth"></d-cite>. Using the same mask with any other random initialization results in a significant drop in performance &mdash; also known as the "sparse training problem".

Our **ICML 2025** paper "Sparse Training from Random Initialization: Aligning Lottery Ticket Masks using Weight Symmetry" <d-cite key="adnan2025sparse"></d-cite> investigates the sparse training problem from a weight-space symmetry perspective and finds:

- **The Problem is Misalignment:** The reason LTH masks don't generalize to new initializations is a misalignment of optimization basins in the loss landscape, which arises from the inherent geometry and permutation symmetries of neural networks. A mask found in one basin won't work well if the new initialization starts in a different, symmetrically equivalent basin.
- **The Solution is Alignment:** We can approximate the permutation that aligns the basins of two different models<d-cite key="ainsworth2023git,jordan2023repair"></d-cite>, and applying this same permutation to the LTH mask, we can successfully train a sparse neural network from a _new_ random initialization. The better the approximation of the permutation, the better the performance.
- **Bridging the Performance Gap:** Training with this **permuted mask** significantly improves generalization compared to naively using the original mask, nearly matching the performance of the original LTH solution across various models and datasets when the models are wide enough to allow accurate permutation matching.
- **Unlocking Diversity:** Unlike standard LTH, which consistently relearns the same solution <d-cite key="evci2022gradientflow"></d-cite>, our permutation-based method can train more diverse solutions when starting from different random initializations.

---

## The Lottery Ticket Hypothesis and the Sparse Training Problem

<div class="container">
  <div class="row justify-content-center align-items-center">
      <div class="col-lg mt-3 mt-md-0 bg-white">
          <img src="/assets/img/sparse-rebasin/sparsetrainingproblem2.svg" alt="Diagram showing the sparse training problem where a pruned mask applied to a new initialization performs poorly." class="img-fluid rounded z-depth-0" loading="eager" />
      </div>
  </div>
  <div class="caption">Figure 1: (Left) The standard pruning pipeline creates a good pruned solution. (Right) The sparse training problem: applying the mask from the pruned solution to a new, different random initialization results in poor performance.</div>
</div>

The quest for smaller, faster, and more efficient neural networks has led to exciting breakthroughs in neural network **sparsity**. One of the most influential ideas in this area is the **Lottery Ticket Hypothesis (LTH)** <d-cite key="frankle2019lth"></d-cite>. LTH suggests that within a large, dense neural network, there are sparse subnetworks (the "winning tickets") that are exceptionally good at training. The standard LTH methodology is:

1.  Train a full, dense neural network.
2.  Prune the connections with the smallest magnitude weights to get a sparse mask.
3.  "Rewind" the weights of the remaining connections to their values from very early in training and train the sparse neural network again.

This process can produce sparse models that match the performance of the original dense one <d-cite key="frankle2019lth"></d-cite>, however requires expensive dense pre-training from many early training checkpoints in practice to identify a "winning ticket". What if we could just use a winning ticket mask to train a sparse model from a _new_ random initialization? This is the heart of the **sparse training problem**. Unfortunately, naively applying this doesn't work well; the performance drops dramatically <d-cite key="frankle2019lth,adnan2025sparse"></d-cite>.

## It's All About Symmetry

<div class="container">
  <div class="row align-items-center justify-content-center">
      <div class="col-10 mt-3 mt-md-0">
          <img src="/assets/img/sparse-rebasin/weightsymmetry3.svg" alt="Diagram illustrating that swapping two neurons results in a functionally identical network." class="img-fluid rounded z-depth-0" loading="eager" />
      </div>
  </div>
  <div class="caption">Figure 2: Swapping two neurons (and their incoming and outgoing weights) results in a functionally identical network, illustrating permutation or  weight symmetry.</div>
</div>

The answer lies in a fundamental property of neural networks: permutation or **weight symmetry**. If you take a layer in a neural network model and swap two of its neurons — including their incoming and outgoing weights — the function the neural network represents remains identical <d-cite key="nielsen1990,entezari2022role"></d-cite>. However, in the high-dimensional space of weight space where we optimize, these two neural network models are at completely different locations.

This symmetry means the loss landscape is filled with many identical, mirrored **loss basins**. When we train a model from a random initialization, it descends into one of these basins. Recent work suggests that most solutions found by SGD lie in a _single_ basin, once you account for these permutations <d-cite key="entezari2022role,ainsworth2023git"></d-cite>.

### The Geometry of the Sparse Training Problem

#### Dense Training and Pruning

<div class="container">
  <div class="row justify-content-center align-items-center">
      <div class="col-lg mt-3 mt-md-0 bg-white">
          <img src="/assets/img/sparse-rebasin/sparsebasin_densepruning.svg" alt="Loss landscape showing dense training and weight magnitude-based pruning." class="img-fluid rounded z-depth-0" loading="eager" />
      </div>
  </div>
  <div class="caption">Figure 3(a): Dense neural network training and pruning; a dense neural network model of only two neurons, each with a single weight $w_0$ and $w_1$ respectively can illustrate the geometry of loss landscapes, and the sparse training problem. Here, dense neural network training and weight magnitude-based pruning results in performant neural network for inference with a sparse mask $\mathbf{m}_A$.</div>
</div>

Here we show the loss landscape of a neural network model of only two neurons, each with a single weight $w_0$ and $w_1$ respectively. The model has two symmetric loss basins/local minima, corresponding to the two possible permutations of the neurons. This simple model can illustrate the geometry of loss landscapes, and convey our intuition about the sparse training problem.

In Figure 3(a) a neural network $A$ is trained from random initialization $\mathbf{w}^{t=0}_A$ to a good solution $\mathbf{w}^{t=T}_A$, and pruned to remove the smallest magnitude weight, defining a mask $\mathbf{m}_A$ and sparse neural network model $\mathbf{w}^{t=T}_A \odot m_A$. In general such dense training and pruning works well, and maintains good generalization (e.g. test accuracy).

#### The Lottery Ticket Hypothesis

<div class="container">
  <div class="row justify-content-center align-items-center">
      <div class="col-lg mt-3 mt-md-0 bg-white">
        <img src="/assets/img/sparse-rebasin/sparsebasin_lth.svg" alt="Loss landscape showing Lottery Ticket Hypothesis methodology." class="img-fluid rounded z-depth-0" loading="eager" />
      </div>
  </div>
  <div class="caption">Figure 3(b): The Lottery Ticket Hypothesis (LTH) suggests the sparse mask $\mathbf{m}_A$ can be trained using the same training procedure as the original model, but with the mask applied from almost the start, achieving sparse training.</div>
</div>

In Figure 3(b) we again train neural network $A$ from the same random initialization $\mathbf{w}^{t=0}_A$ however, in this case we train sparse, i.e. using the mask $\mathbf{m}_A$. This is the equivalent of projecting our initial weights down to the subspace defined by the mask, in this case the single dimension $\mathbf{w}_0$, and training in that restricted subspace, i.e. in this case along the one-dimensional subspace aligned with $\mathbf{w}_0$. We still find a good solution $\mathbf{w}^{t=T}_A \odot \mathbf{m}_A$ which maintains good generalization (e.g. test accuracy).

This is what the Lottery Ticket Hypothesis (LTH) suggests: that the sparse mask $\mathbf{m}_A$ can be trained using the same training procedure as the original model, but with the mask applied from almost the start, achieving sparse training.

#### The Sparse Training Problem

<div class="container">
  <div class="row justify-content-center align-items-center">
      <div class="col-lg mt-3 mt-md-0 bg-white">
          <img src="/assets/img/sparse-rebasin/sparsebasin_sparsetrainingproblem.svg" alt="Loss landscape showing dense training and weight magnitude-based pruning." class="img-fluid rounded z-depth-0" loading="eager" />
      </div>
  </div>
  <div class="caption">Figure 3(c): The sparse training problem is illustrated by attempting to train model $B$ from a new random initialization, $\mathbf{w}^{t=0}_B$, while re-using the sparse mask $m_A$ discovered from pruning. Sparse training of model $B$ fails with the original mask, as one of the most important weights is not preserved.</div>
</div>

Finally, in Figure 3(c) we illustrate the sparse training problem. Here we attempt to train a new neural network $B$ from a new random initialization, $\mathbf{w}^{t=0}_B$, while re-using the sparse mask $\mathbf{m}_A$ discovered from pruning neural network $A$. Sparse training of neural network $B$ fails with the original mask, as one of the most important weights is not preserved when we project using $\mathbf{m}_A$, projecting our weight instead to a location far from the solution basin, and leading to poor generalization (e.g. test accuracy).

### The Hypothesis: A Tale of Two Basins

<div class="container">
  <div class="row justify-content-center align-items-center">
      <div class="col-lg mt-3 mt-md-0 bg-white">
          <img src="/assets/img/sparse-rebasin/sparsebasin_permuted.svg" alt="Loss landscape showing dense training and weight magnitude-based pruning." class="img-fluid rounded z-depth-0" loading="eager" />
      </div>
  </div>
  <div class="caption">Figure 3(d): Our solution is to permute the mask to $\pi(m_A)$, which aligns with model B's basin and enables successful sparse training (green path).The permuted mask $\pi(m_A)$ aligns with model B's basin, enabling successful sparse training (green path).</div>
</div>

This brings us to our core hypothesis <d-cite key="adnan2025sparse"></d-cite>:
**An LTH mask fails on a new initialization because the mask is aligned to one basin, while the new random initialization has landed in another.**

The optimization process is essentially starting in the wrong valley for the map it's been given. Naively applying the mask pulls the new initialization far away from a good solution path, leading to poor performance <d-cite key="adnan2025sparse"></d-cite>.

**But what if we could "rotate" the mask to match the orientation of the new basin?** This is exactly what we propose.

---

## The Method: Aligning Masks with Permutations

Our method leverages recent advances in model merging, like Git Re-Basin <d-cite key="ainsworth2023git"></d-cite>, which find the permutation that aligns the neurons of two separately trained models. Our training paradigm is as follows <d-cite key="adnan2025sparse"></d-cite>:

1.  **Train Two Dense Models:** Start with two different random initializations, $\mathbf{w}_A^{t=0}$ and $\mathbf{w}_B^{t=0}$, and train them to convergence to get two dense models, $\mathbf{w}_A^{t=T}$ and $\mathbf{w}_B^{t=T}$, or `Model A` and `Model B`.
    <div class="container">
      <div class="row justify-content-center align-items-center">
        <div class="col-lg mt-3 mt-md-0 bg-white">
            <img src="/assets/img/sparse-rebasin/method_step1.svg" alt="Method Step 1: Dense Training of Two models." class="img-fluid rounded z-depth-0" loading="eager" />
        </div>
      </div>
    </div>
2.  **Get the LTH Mask:** Prune `Model A` using standard iterative magnitude pruning (IMP) to get a sparse "winning ticket" mask, $\mathbf{m}_A$.
    <div class="container">
      <div class="row justify-content-center align-items-center">
        <div class="col-lg mt-3 mt-md-0 bg-white">
            <img src="/assets/img/sparse-rebasin/method_step2.svg" alt="Method Step 1: Dense Training of Two models." class="img-fluid rounded z-depth-0" loading="eager" />
        </div>
      </div>
    </div>
3.  **Find the Permutation relating the Models:** Use an **activation matching** algorithm <d-cite key="jordan2023repair"></d-cite> to find the permutation, $\pi$, that best aligns the neurons of `Model A` with `Model B`, i.e. $\mathbf{w}_B^{t=T} = \pi(\mathbf{w}_A^{t=T})$. This essentially finds the "rotation" or permutation needed to map one solution basin onto the other.
    <div class="container">
      <div class="row justify-content-center align-items-center">
        <div class="col-lg mt-3 mt-md-0 bg-white">
            <img src="/assets/img/sparse-rebasin/method_step3.svg" alt="Method Step 1: Dense Training of Two models." class="img-fluid rounded z-depth-0" loading="eager" />
        </div>
      </div>
    </div>
4.  **Permute the Mask:** Apply the permutation $\pi$ to the mask $\pi(\mathbf{m}_A)$ for `Model A` to get a new, aligned mask: $\mathbf{m}_B = \pi(\mathbf{m}_A)$ for `Model B`.
    <div class="container">
      <div class="row justify-content-center align-items-center">
        <div class="col-lg mt-3 mt-md-0 bg-white">
          <img src="/assets/img/sparse-rebasin/method_step4.svg" alt="Method Step 1: Dense Training of Two models." class="img-fluid rounded z-depth-0" loading="eager" />
        </div>
      </div>
    </div>
5.  **Train from Scratch (Almost)!** Train a new sparse model starting from the $\mathbf{w}_B$ initialization (rewound to an early checkpoint, $k$), but using the **permuted mask** $\pi(\mathbf{m}_A)$.
<div class="container">
<div class="row justify-content-center align-items-center">
<div class="col-lg mt-3 mt-md-0 bg-white">
<img src="/assets/img/sparse-rebasin/method_step5.svg" alt="Method Step 1: Dense Training of Two models." class="img-fluid rounded z-depth-0" loading="eager" />
</div>
</div>
</div>

### Permutated vs. the LTH and Naive Baselines

<div class="container l-screen">
  <div class="row justify-content-center align-items-center">
      <div class="col-lg mt-3 mt-md-0 bg-white">
          <img src="/assets/img/sparse-rebasin/methodology.svg" alt="Diagram of the training paradigm, from training dense models to permutation matching and final sparse training." class="img-fluid rounded z-depth-0" loading="eager" />
      </div>
  </div>
  <div class="caption">Figure 4: The overall framework of our training procedure <d-cite key="adnan2025sparse"></d-cite>. We use two trained dense models to find a permutation $\pi$. This permutation is then applied to the mask from Model A, allowing it to be successfully used to train Model B from a random initialization.</div>
</div>

We present the methodology of the three different training paradigms we compare in our results here in Figure 4:

1. **LTH**: The original Lottery Ticket Hypothesis approach, which trains a dense model and then prunes it.
2. **Naive**: A straightforward application of the pruned mask from Model A to Model B without any permutation, this is the standard sparse training problem setup, and performs poorly.
3. **Permuted**: Our proposed method, which finds a permutation of the weights to better align the two models.

---

## The Result: It Works (Approximately!)

Across a wide range of experiments, our method demonstrates that aligning the mask is the key to solving the sparse training problem for LTH masks <d-cite key="adnan2025sparse"></d-cite>. Of course the permutation matching is only approximate, and so the performance doesn't perfectly match the original LTH solution, but it comes remarkably close, especially as model width increases which has been shown to improve permutation matching quality <d-cite key="ainsworth2023git,jordan2023repair"></d-cite>.

### Closing the Performance Gap

When we compare the performance of the standard `LTH` solution, the `Naive` solution (un-permuted mask on a new init), and our `Permuted` solution, the results are clear. The `Permuted` approach consistently and significantly outperforms the `Naive` baseline, closing most of the performance gap to the original `LTH` solution. The effect is especially pronounced at higher sparsity levels, where the `Naive` method struggles most <d-cite key="adnan2025sparse"></d-cite>.

<div class="container l-screen">
  <div class="row justify-content-center align-items-center">
      <div class="col-lg mt-3 mt-md-0 bg-white">
          <img src="/assets/img/sparse-rebasin/results_cifar10.svg" alt="Graphs showing test accuracy vs rewind points for ResNet20 on CIFAR-10 at different sparsity levels and widths." class="img-fluid rounded z-depth-0" loading="eager" />
      </div>
  </div>
  <div class="caption">Figure 5: Test accuracy on CIFAR-10 for ResNet20 of varying widths (`w`) and sparsities. The permuted solution (blue) consistently outperforms the naive one (orange) and gets closer to the LTH baseline (green), especially as model width increases <d-cite key="adnan2025sparse"></d-cite>.</div>
</div>

### Wider is Better, and More Diverse

We found that our method works even better on wider models. Wider networks have smoother loss landscapes, which allows the activation matching algorithm to find a more accurate permutation, reducing the loss barrier between basins <d-cite key="adnan2025sparse"></d-cite>.

<div class="container l-screen">
  <div class="row justify-content-center align-items-center">
      <div class="col-lg mt-3 mt-md-0 bg-white">
          <img src="/assets/img/sparse-rebasin/results_cifar100.svg"  alt="Table showing ensemble diversity metrics for different sparse models." class="img-fluid rounded z-depth-0" loading="eager" />
      </div>
  </div>
  <div class="caption">Figure 6: Test accuracy on CIFAR-100 for ResNet20 of varying widths (`w`) and sparsities. The permuted solution (blue) consistently outperforms the naive one (orange) and gets closer to the LTH baseline (green), especially as model width increases <d-cite key="adnan2025sparse"></d-cite>.</div>
</div>

Furthermore, by starting from different random initializations, our method can produce a more **diverse set of solutions** compared to LTH, which is known to be functionally very similar to the pruned model it came from <d-cite key="evci2022gradientflow"></d-cite>. This diversity is beneficial: an ensemble of our `permuted` models achieves significantly higher accuracy than an ensemble of `LTH` models, which are too similar to provide much of a boost <d-cite key="adnan2025sparse"></d-cite>.

<div class="container l-screen">
  <div class="row justify-content-center align-items-center">
      <div class="col-lg mt-3 mt-md-0 bg-white">
          <img src="/assets/img/sparse-rebasin/results_vgg.svg"  alt="Table showing ensemble diversity metrics for different sparse models." class="img-fluid rounded z-depth-0" loading="eager" />
      </div>
  </div>
  <div class="caption">Figure 7: Test accuracy on CIFAR-10 for VGG-11 of varying widths (`w`) and sparsities. The permuted solution (blue) consistently outperforms the naive one (orange) and gets closer to the LTH baseline (green), especially as model width increases <d-cite key="adnan2025sparse"></d-cite>.
</div>

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
