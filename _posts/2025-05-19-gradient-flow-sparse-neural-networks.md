---
layout: distill
title: "Gradient Flow in Sparse Neural Networks & Why Lottery Tickets (Sometimes) Win"
description: "An exploration of why sparse neural networks are hard to train and how understanding gradient flow sheds light on Lottery Tickets and Dynamic Sparse Training."
date: 2022-02-24
authors:
  - name: Utku Evci
    url: "https://scholar.google.com/citations?user=SpM0R6AAAAAJ"
    affiliations:
      name: Google
  - name: Yani Ioannou
    url: "https://yani.ai/"
    affiliations:
      name: University of Calgary
  - name: Cem Keskin
    url: "https://scholar.google.com/citations?user=L4t47LAAAAAJ"
    affiliations:
      name: Facebook
  - name: Yann Dauphin
    url: "https://scholar.google.com/citations?user=dM0qAN4AAAAAJ"
    affiliations:
      name: Google
paper_url: https://arxiv.org/abs/2010.03533
bibliography: 2025-05-19-gradient-flow-sparse-neural-networks.bib

toc: true
related_posts: true
---

## TL;DR

Training sparse neural networks directly from a random initialization is notoriously difficult, often resulting in poor performance compared to their dense counterparts. The paper "Gradient Flow in Sparse Neural Networks and How Lottery Tickets Win" <d-cite key="Evci2022GradientFlow"></d-cite>, accepted for an [Oral Presentation at AAAI 2022](https://aaai-2022.virtualchair.net/poster_aaai3082), investigates this through the perspective of gradient flow and finds:

- **Poor Gradient Flow at Initialization:** Standard initialization techniques, designed for dense networks, are ill-suited for sparse networks due to their heterogeneous connectivity. This leads to vanishing gradients right from the start. We propose a sparsity-aware initialization that can alleviate this.
- **Poor Gradient Flow During Training:** Even if initialized better, sparse networks can suffer from weak gradient flow throughout training. Dynamic Sparse Training (DST) methods, which adapt network connectivity during training, can significantly improve this.
- **Lottery Tickets Re-learn, Don't Magically Fix Flow:** The success of Lottery Tickets (LTs) isn't due to them inherently having better gradient flow. Instead, LTs (which use specific initial weights from a pre-trained dense model's history) effectively "re-learn" the good solution found by pruning the original dense model. They are guided to a known good basin of attraction, rather than finding a new one through superior optimization dynamics in a sparse setting.

## The Quest for Efficient Yet Powerful Neural Networks

Deep Neural Networks (DNNs) are the powerhouses behind many AI breakthroughs. However, their increasing size and computational appetite pose significant challenges for deployment and training sustainability. One promising avenue for efficiency is **sparsity**: using networks with far fewer connections (and thus parameters) than typical dense networks.

A common way to obtain a sparse network is by **pruning** a large, trained dense network. This often yields sparse models that retain the performance of the original dense model with a fraction of the parameters.

<div class="container">
  <div class="row justify-content-center align-items-center">
      <div class="col-lg mt-3 mt-md-0 bg-white">
          <img src="/assets/img/gradientflow/sparsetrainingproblem.svg" alt="Training Outcomes: Pruning pipeline vs. sparse training problem." class="img-fluid rounded z-depth-0" loading="eager" />
      </div>
  </div>
  <div class="caption">Figure 1: (Left) The standard pruning pipeline: train a dense model, prune it, and optionally fine-tune to get a good sparse model. (Right) The sparse training problem: initializing a sparse network randomly and training it often leads to poor performance compared to the pruned model.</div>
</div>

However, what if we want to train a sparse network from the get-go, without the costly pre-training of a dense model? This is where things get tricky. Naively initializing a network with a sparse structure and training it from scratch (the "sparse training problem") usually leads to significantly worse performance. This begs the question: why is training sparse networks so hard, and what can we learn from the exceptions?

## The Importance of Gradient Flow

Many advancements in training dense DNNs have come from understanding and improving **gradient flow** – how the error signals propagate backward through the network to update the weights. Poor gradient flow can lead to vanishing or exploding gradients, making training stall or become unstable. This paper <d-cite key="Evci2022GradientFlow"></d-cite> applies this lens to sparse neural networks.

## Problem 1: Off to a Bad Start &mdash; Poor Gradient Flow at Initialization

### Dense Initialization and Sparse Fan-in

<div class="container">
  <div class="row align-items-center justify-content-center">
      <div class="col-6 mt-3 mt-md-0">
          <img src="/assets/img/gradientflow/dense_fanin.svg" alt="Dense fan-in in a neural network." class="img-fluid rounded z-depth-0" loading="eager" />
          <div class="caption">(a) Dense layer, every neuron has the same number of incoming connections</div>
      </div>
      <div class="col-6 mt-3 mt-md-0">
          <img src="/assets/img/gradientflow/sparse_fanin.svg" alt="Heterogeneous sparse fan-in in a neural network." class="img-fluid rounded z-depth-0" loading="eager" />
          <div class="caption">(b) Sparse layer, every neuron can have a different number of incoming connections</div>
      </div>
  </div>
  <!-- Dense vs Sparse Fan-in: Illustrating differing fan-in for dense and sparse layers -->
  <div class="caption">Figure: (a) In a dense layer, each neuron has the same fan-in, (b) However, in a general unstructured sparse layer, the fan-in can vary significantly from neuron to neuron &mdash; we propose an initialization that accounts for this.</div>
</div>

Standard weight initialization methods like Glorot/He <d-cite key="Glorot2010Understanding,He2015Delving"></d-cite> are designed with dense networks in mind. They assume that all neurons in a layer have the same number of incoming (fan-in) and outgoing (fan-out) connections.

For example, for a layer using a ReLU activation function, the weights are initialized from a distribution with variance inversely proportional to the largest number of incoming connections (fan-in):

$$w_{ij}^{[l]} \sim \mathcal{N}(0, \frac{2}{\text{fan-in}}), $$

where $\textbf{fan-in}$ is the number of incoming connections for the layer, and $w_{ij}^{[l]}$ is the weight connecting neuron $i$ in layer $l-1$ to neuron $j$ in layer $l$. This ensures that the variance of the output of the layer is roughly equal to the variance of its input, which helps maintain a healthy signal flow through the network.

### A Sparsity-Aware Initialization to Fix Gradient Flow at Initialization

<div class="container">
  <div class="row align-items-center justify-content-center">
      <div class="col-6 mt-3 mt-md-0">
          <img src="/assets/img/gradientflow/sparse_fanin_unequalout.svg" alt="Sparse neural network with dense init." class="img-fluid rounded z-depth-0" loading="eager" />
          <div class="caption">(a) Dense initialization assumes every neuron has same number of connections, and on average, uses weights that are too small</div>
      </div>
      <div class="col-6 mt-3 mt-md-0">
          <img src="/assets/img/gradientflow/sparse_fanin_equalout.svg" alt="Sparse neural network with sparse init.." class="img-fluid rounded z-depth-0" loading="eager" />
          <div class="caption">(b) Sparse initialization calculates the correct weight variance for each neuron based on the number of incoming connections</div>
      </div>
  </div>
  <!-- Dense vs Sparse Fan-in: Illustrating differing fan-in for dense and sparse layers -->
  <div class="caption">Figure: (a) Using a dense initialization for a sparse layer causes vanishing gradients as neurons with few connections are initialized incorrectly, however in (b) the sparse initialization accounts for the fact that fan-in can vary significantly from neuron to neuron &mdash; ensuring better behaved gradients at initialization.</div>
</div>

In a sparse neural network, the assumption that every neuron has the same number of connections breaks down. Rather, the number of connections per neuron can be highly variable. Using dense initializations directly in sparse networks often causes the signal to vanish rapidly as it propagates through the layers. This is because neurons with fewer incoming connections are initialized with weights that are too small, leading to a weak signal. This can cause the gradients to vanish, especially in deeper networks, making it hard for the model to learn effectively.

Our work <d-cite key="Evci2022GradientFlow"></d-cite> proposes a **sparsity-aware initialization** that adjusts the variance of the initial weights for each neuron based on its _actual_ fan-in within the sparse structure, for example for a layer with a ReLU activation function:

$$w_{ij}^{[l]} \sim \mathcal{N}(0, \frac{2}{\text{fan-in}_i^{[l]}}), $$

where $\text{fan-in}_i^{[l]}$ is the number of incoming connections for neuron $i$ in layer $l$.

### Signal Propagation at Initialization with Different Initializations

<div class="container">
  <div class="row justify-content-center align-items-center">
      <div class="col-8 mt-3 mt-md-0 bg-white">
          <img src="/assets/img/gradientflow/sparseinit_signalprop.svg" alt="Signal Propagation at Initialization: Graph showing standard deviation of output vs. sparsity." class="img-fluid rounded z-depth-0" loading="eager" />
      </div>
  </div>
  <div class="caption">Figure 3: Standard deviation of the pre-softmax output ($\sigma(f(x))$) in LeNet-5 vs. sparsity level. Dense initialization (blue) shows signal vanishing with increasing sparsity. Sparsity-aware initializations (Liu et al. <d-cite key="Liu2019Rethinking"></d-cite> and "Ours" - the paper's proposal) maintain signal strength.</div>
</div>

This sparsity-aware initialization leads to better signal propagation at the start of training and can improve the final generalization performance, especially for networks without normalization layers like BatchNorm (e.g., LeNet5, VGG16). For models with BatchNorm (e.g., ResNet-50), the effect of initialization is less pronounced, as BatchNorm itself helps regulate signal propagation.

## Problem 2: Slogging Through &mdash; Poor Gradient Flow During Training

While a good initialization helps, it's not the whole story. Sparse networks can still suffer from poor gradient flow _during_ the training process.

<div class="container">
  <div class="row justify-content-center align-items-center bg-white">
      <div class="col-6 mt-3 mt-md-0">
          <img src="/assets/img/gradientflow/mnist_gradnorm_logx.svg" alt="Gradient Norm During Training: Graphs for LeNet-5, VGG-16, and ResNet-50." class="img-fluid rounded z-depth-0" loading="eager" />
      </div>
      <div class="col-6 mt-3 mt-md-0">
          <img src="/assets/img/gradientflow/vgg_gradnorm.svg" alt="Gradient Norm During Training: Graphs for LeNet-5, VGG-16, and ResNet-50." class="img-fluid rounded z-depth-0" loading="eager" />
      </div>
      <div class="col-6 mt-3 mt-md-0">
          <img src="/assets/img/gradientflow/resnet_gradnorm.svg" alt="Gradient Norm During Training: Graphs for LeNet-5, VGG-16, and ResNet-50." class="img-fluid rounded z-depth-0" loading="eager" />
      </div>
  </div>
  <div class="caption">Figure 4: Gradient norm during training for LeNet-5 (left), VGG-16 (center), and ResNet-50 (right) under different setups. 'Scratch' (training a sparse mask from random dense initialization) often shows very low gradient norm initially. 'Scratch+' (with sparsity-aware initialization) improves this. 'RigL+' (a DST method with sparsity-aware init) often shows stronger gradient flow.</div>
</div>

This is where **Dynamic Sparse Training (DST)** methods come in. DST techniques, like RigL <d-cite key="Evci2020RigL"></d-cite>, don't keep the sparse connectivity fixed. Instead, they periodically update the mask during training:

1.  **Prune:** Remove connections that have become less salient (e.g., small magnitude weights).
2.  **Grow:** Add new connections, often by identifying those that would have the largest gradient if they were active.

The paper shows that DST methods, particularly RigL, significantly improve gradient flow during training compared to training with a fixed sparse mask. These updates can introduce new directions for optimization (e.g., by creating new negative eigenvalues in the Hessian), helping the network escape poor regions of the loss landscape. This improved gradient flow correlates with better generalization performance.

## The Curious Case of Lottery Tickets (LTs)

The Lottery Ticket Hypothesis (LTH) <d-cite key="Frankle2019LTH"></d-cite> proposed that within a large, randomly initialized dense network, there exist smaller subnetworks (the "winning tickets"). If these winning tickets are trained in isolation from their _original initialization weights_ (or weights from very early in the dense model's training, known as "late rewinding"), they can achieve accuracy comparable to the full dense network.

<div class="container">
  <div class="row justify-content-center align-items-center">
      <div class="col-lg mt-3 mt-md-0 bg-white">
          <img src="/assets/img/gradientflow/lotterticketsolution.svg" alt="Lottery Ticket Hypothesis Concept: Diagram illustrating the LTH process." class="img-fluid rounded z-depth-0" loading="eager" />
      </div>
  </div>
  <div class="caption">Figure 5: The Lottery Ticket Hypothesis: A dense network is trained (obtaining a dense solution), then pruned. The "winning ticket" uses the *initial weights* ($\Theta_{t=0}$ or an early snapshot $\Theta_{0<t \ll T}$) corresponding to the pruned mask and is then trained.</div>
</div>

This was exciting because it suggested a way to find highly sparse, trainable networks. However, the paper <d-cite key="Evci2022GradientFlow"></d-cite> finds something intriguing:
**Lottery Tickets also exhibit poor gradient flow, similar to naively trained sparse networks!** (see Figure 4).

So, if LTs don't fix the gradient flow problem, why do they work so well? The paper's central argument is that LTs succeed because they essentially **re-learn the pruning solution** they were derived from.

### Evidence for LTs Re-learning the Pruning Solution

<div class="container">
  <div class="row justify-content-center align-items-center">
      <div class="col-9 mt-3 mt-md-0 bg-white">
          <img src="/assets/img/gradientflow/mnist_mds.svg" alt="MDS Plot of Solutions: 2D projection of solution distances for LeNet5." class="img-fluid rounded z-depth-0" loading="eager" />
      </div>
  </div>
  <div class="caption">Figure 6: A 2D MDS projection showing the relative distances between different solutions for LeNet5. 'Lottery-start' is closer to 'Prune-end' than 'Scratch-start'. 'Lottery-end' converges very close to 'Prune-end', while 'Scratch-end' solutions are more dispersed and further away.</div>
</div>

1.  **Proximity in Weight Space:**
    LT initializations (the specific weight values rewound from early in dense training) start much closer in L2 distance to the final _pruned solution_ (the weights of the dense model after pruning) than a random "scratch" initialization using the same mask. After training, the LT solution ends up significantly closer to this pruned solution.

<div class="container">
  <div class="row justify-content-center align-items-center">
      <div class="col-6 mt-3 mt-md-0 bg-white">
          <img src="/assets/img/gradientflow/mnist_interpol.svg" alt="Loss Interpolation: Graph showing training loss along interpolation paths for LeNet5." class="img-fluid rounded z-depth-0" loading="eager" />
      </div>
      <div class="col-6 mt-3 mt-md-0 bg-white">
          <img src="/assets/img/gradientflow/resnet_interpol.svg" alt="Loss Interpolation: Graph showing training loss along interpolation paths for LeNet5." class="img-fluid rounded z-depth-0" loading="eager" />
      </div>
  </div>
  <div class="caption">Figure 7: Training loss along a linear interpolation path between a starting point ($\alpha=0$, e.g., Lottery-start or Scratch-start) and the Pruned Solution ($\alpha=1$) for LeNet5. The path between 'Lottery End' and 'Pruned Solution' is relatively flat, indicating they are in the same basin. The path from 'Scratch End' often shows a barrier.</div>
</div>

{:start="2"}

1.  **Same Basin of Attraction:**
    By interpolating between the LT solution/initialization and the pruned solution, the paper shows that they lie within the same low-loss basin of attraction. In contrast, scratch solutions often have a high loss barrier separating them from the pruned solution's basin.

{:start="3"}

1.  **Functional Similarity:**
    LT solutions are not only close in weight space but also learn very similar functions to the pruned solution they originated from. This is measured by low "disagreement" (fraction of test images classified differently) between the LT solution and the pruned solution. Ensembles of LTs derived from the same pruning process show little performance gain, further suggesting they converge to nearly identical functions.

### The Lottery Ticket Initialization: A Nudge in the Right Direction

<div class="container l-screen">
  <div class="row justify-content-center align-items-center">
      <div class="col-lg mt-3 mt-md-0 bg-white">
          <img src="/assets/img/gradientflow/initializations_explained.svg" alt="Loss Landscape Intuition: Diagram illustrating basins of attraction for LT and Scratch." class="img-fluid rounded z-depth-0" loading="eager" />
      </div>
  </div>
  <div class="caption">Figure 8: An intuitive illustration. A Lottery Ticket initialization (blue circle) is already positioned within the basin of attraction of the good Pruning Solution (green circle). Random (Scratch) initializations (red circles) are more likely to fall into different, potentially suboptimal, basins.</div>
</div>
**The implication is powerful:** LTs aren't discovering new, highly effective sparse configurations through superior optimization dynamics. Instead, their specific initialization "nudges" the optimization process to rediscover a known good solution – the one found by pruning the dense network.

## Key Insights Summarized

This investigation into gradient flow in sparse neural networks reveals:

1.  **Sparsity-Aware Initialization Matters:** Naive use of dense initializations harms sparse networks by causing poor gradient flow from the start. Using initializations that account for the actual sparse connectivity is crucial.
2.  **Dynamic Sparse Training Boosts Gradient Flow:** DST methods improve gradient flow _during_ training by adapting the network's sparse connections, leading to better generalization than training with fixed sparse masks.
3.  **Lottery Tickets are "Echoes" of Pruning:** LTs work well not because they inherently possess better gradient flow, but because their specific initial weights guide them to re-learn the solution of the pruned dense model they originated from. This limits their ability to find truly novel solutions in the sparse regime.

## Conclusion and Future Directions

Understanding gradient flow provides valuable insights into the challenges of training sparse neural networks. While sparsity-aware initializations and Dynamic Sparse Training offer promising avenues for improving how we train sparse models from scratch, the success of Lottery Tickets seems more about "remembering" a good solution than fundamentally solving the optimization difficulties in sparse landscapes.

The journey towards efficiently training sparse neural networks that are as performant as their dense counterparts, without relying on dense pre-training or specific "winning ticket" initializations, continues. Methods that can robustly navigate the complex loss landscapes of sparse models and maintain healthy gradient flow are key to unlocking the full potential of sparse AI.

{% twitter https://x.com/yanii/status/1496532190682927107 %}
