---
title: "Untangling Gradient Flow in Sparse Neural Networks & Why Lottery Tickets (Sometimes) Win"
description: "An exploration of why sparse neural networks are hard to train and how understanding gradient flow sheds light on Lottery Tickets and Dynamic Sparse Training."
date: 2025-05-19
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
bibliography: references.bib # Assuming you'll use a references.bib file
---

## TL;DR

Training sparse neural networks directly from a random initialization is notoriously difficult, often resulting in poor performance compared to their dense counterparts. The paper "Gradient Flow in Sparse Neural Networks and How Lottery Tickets Win" <d-cite key="Evci2022GradientFlow"></d-cite> investigates this through the perspective of gradient flow and finds:

* **Poor Gradient Flow at Initialization:** Standard initialization techniques, designed for dense networks, are ill-suited for sparse networks due to their heterogeneous connectivity. This leads to vanishing gradients right from the start. Sparsity-aware initializations can alleviate this.
* **Poor Gradient Flow During Training:** Even if initialized better, sparse networks can suffer from weak gradient flow throughout training. Dynamic Sparse Training (DST) methods, which adapt network connectivity during training, can significantly improve this.
* **Lottery Tickets Re-learn, Don't Magically Fix Flow:** The success of Lottery Tickets (LTs) isn't due to them inherently having better gradient flow. Instead, LTs (which use specific initial weights from a pre-trained dense model's history) effectively "re-learn" the good solution found by pruning the original dense model. They are guided to a known good basin of attraction, rather than finding a new one through superior optimization dynamics in a sparse setting.

## The Quest for Efficient Yet Powerful Neural Networks

Deep Neural Networks (DNNs) are the powerhouses behind many AI breakthroughs. However, their increasing size and computational appetite pose significant challenges for deployment and training sustainability. One promising avenue for efficiency is **sparsity**: using networks with far fewer connections (and thus parameters) than typical dense networks.

A common way to obtain a sparse network is by **pruning** a large, trained dense network. This often yields sparse models that retain the performance of the original dense model with a fraction of the parameters.

![Training Outcomes](placeholder_figure_training_outcomes.png)
*Figure 1: (Left) The standard pruning pipeline: train a dense model, prune it, and optionally fine-tune to get a good sparse model. (Right) The sparse training problem: initializing a sparse network randomly and training it often leads to poor performance compared to the pruned model. (Adapted from Slide 2/3 of the presentation)*

However, what if we want to train a sparse network from the get-go, without the costly pre-training of a dense model? This is where things get tricky. Naively initializing a network with a sparse structure and training it from scratch (the "sparse training problem") usually leads to significantly worse performance. This begs the question: why is training sparse networks so hard, and what can we learn from the exceptions?

## The Importance of Gradient Flow

Many advancements in training dense DNNs have come from understanding and improving **gradient flow** – how the error signals propagate backward through the network to update the weights. Poor gradient flow can lead to vanishing or exploding gradients, making training stall or become unstable. This paper <d-cite key="Evci2022GradientFlow"></d-cite> applies this lens to sparse neural networks.

## Problem 1: Off to a Bad Start – Poor Gradient Flow at Initialization

Standard weight initialization methods like Glorot/He <d-cite key="Glorot2010Understanding"></d-cite> <d-cite key="He2015Delving"></d-cite> are designed with dense networks in mind. They assume that all neurons in a layer have roughly the same number of incoming (fan-in) and outgoing (fan-out) connections.

![Dense vs Sparse Fan-in](placeholder_figure_fan_in.png)
*Figure 2: (a) In a dense layer, each neuron has the same fan-in. (b) In an unstructured sparse layer, the fan-in can vary significantly from neuron to neuron. Standard initializations don't account for this. (Adapted from Slide 8 of the presentation / Fig 1a,b of the paper)*

In an unstructured sparse network, this assumption breaks down. The number of connections per neuron can be highly variable. Using dense initializations directly in sparse networks often causes the signal to vanish rapidly as it propagates through the layers.

The paper proposes a **sparsity-aware initialization** that adjusts the variance of the initial weights for each neuron based on its *actual* fan-in within the sparse structure:
$w_{ij}^{[l]} \sim \mathcal{N}(0, \frac{2}{\text{fan-in}_i^{[l]}})$
where $\text{fan-in}_i^{[l]}$ is the number of incoming connections for neuron $i$ in layer $l$.

![Signal Propagation at Initialization](placeholder_figure_signal_propagation_init.png)
*Figure 3: Standard deviation of the pre-softmax output ($\sigma(f(x))$) in LeNet-5 vs. sparsity level. Dense initialization (blue) shows signal vanishing with increasing sparsity. Sparsity-aware initializations (Liu et al. <d-cite key="Liu2019Rethinking"></d-cite> and "Ours" - the paper's proposal) maintain signal strength. (Adapted from Slide 10 of the presentation / Fig 1c of the paper)*

This sparsity-aware initialization leads to better signal propagation at the start of training and can improve the final generalization performance, especially for networks without normalization layers like BatchNorm (e.g., LeNet5, VGG16). For models with BatchNorm (e.g., ResNet-50), the effect of initialization is less pronounced, as BatchNorm itself helps regulate signal propagation.

## Problem 2: Slogging Through – Poor Gradient Flow During Training

While a good initialization helps, it's not the whole story. Sparse networks can still suffer from poor gradient flow *during* the training process.

![Gradient Norm During Training](placeholder_figure_gradient_norm_training.png)
*Figure 4: Gradient norm during training for LeNet-5 (left), VGG-16 (center), and ResNet-50 (right) under different setups. 'Scratch' (training a sparse mask from random dense initialization) often shows very low gradient norm initially. 'Scratch+' (with sparsity-aware initialization) improves this. 'RigL+' (a DST method with sparsity-aware init) often shows stronger gradient flow. (Adapted from Slide 11 of the presentation / Fig 2 of the paper)*

This is where **Dynamic Sparse Training (DST)** methods come in. DST techniques, like RigL <d-cite key="Evci2020RigL"></d-cite>, don't keep the sparse connectivity fixed. Instead, they periodically update the mask during training:
1.  **Prune:** Remove connections that have become less salient (e.g., small magnitude weights).
2.  **Grow:** Add new connections, often by identifying those that would have the largest gradient if they were active.

The paper shows that DST methods, particularly RigL, significantly improve gradient flow during training compared to training with a fixed sparse mask. These updates can introduce new directions for optimization (e.g., by creating new negative eigenvalues in the Hessian), helping the network escape poor regions of the loss landscape. This improved gradient flow correlates with better generalization performance.

## The Curious Case of Lottery Tickets (LTs)

The Lottery Ticket Hypothesis (LTH) <d-cite key="Frankle2019LTH"></d-cite> proposed that within a large, randomly initialized dense network, there exist smaller subnetworks (the "winning tickets"). If these winning tickets are trained in isolation from their *original initialization weights* (or weights from very early in the dense model's training, known as "late rewinding"), they can achieve accuracy comparable to the full dense network.

![Lottery Ticket Hypothesis Concept](placeholder_figure_lth_concept.png)
*Figure 5: The Lottery Ticket Hypothesis: A dense network is trained (obtaining a dense solution), then pruned. The "winning ticket" uses the *initial weights* ($\Theta_{t=0}$ or an early snapshot $\Theta_{0<t \ll T}$) corresponding to the pruned mask and is then trained. (Adapted from Slide 16/17 of the presentation)*

This was exciting because it suggested a way to find highly sparse, trainable networks. However, the paper <d-cite key="Evci2022GradientFlow"></d-cite> finds something intriguing:
**Lottery Tickets also exhibit poor gradient flow, similar to naively trained sparse networks!** (See "Lottery" lines in Figure 4).

So, if LTs don't fix the gradient flow problem, why do they work so well? The paper's central argument is that LTs succeed because they essentially **re-learn the pruning solution** they were derived from.

### Evidence for LTs Re-learning the Pruning Solution

1.  **Proximity in Weight Space:**
    LT initializations (the specific weight values rewound from early in dense training) start much closer in L2 distance to the final *pruned solution* (the weights of the dense model after pruning) than a random "scratch" initialization using the same mask. After training, the LT solution ends up significantly closer to this pruned solution.

    ![MDS Plot of Solutions](placeholder_figure_mds_solutions.png)
    *Figure 6: A 2D MDS projection showing the relative distances between different solutions for LeNet5. 'Lottery-start' is closer to 'Prune-end' than 'Scratch-start'. 'Lottery-end' converges very close to 'Prune-end', while 'Scratch-end' solutions are more dispersed and further away. (Adapted from Slide 22 of the presentation / Fig 5a of the paper)*

2.  **Same Basin of Attraction:**
    By interpolating between the LT solution/initialization and the pruned solution, the paper shows that they lie within the same low-loss basin of attraction. In contrast, scratch solutions often have a high loss barrier separating them from the pruned solution's basin.

    ![Loss Interpolation](placeholder_figure_loss_interpolation.png)
    *Figure 7: Training loss along a linear interpolation path between a starting point ($\alpha=0$, e.g., Lottery-start or Scratch-start) and the Pruned Solution ($\alpha=1$) for LeNet5. The path between 'Lottery End' and 'Pruned Solution' is relatively flat, indicating they are in the same basin. The path from 'Scratch End' often shows a barrier. (Adapted from Slide 24 of the presentation / Fig 5c of the paper)*

    ![Loss Landscape Intuition](placeholder_figure_loss_landscape.png)
    *Figure 8: An intuitive illustration. A Lottery Ticket initialization (blue circle) is already positioned within the basin of attraction of the good Pruning Solution (green circle). Random (Scratch) initializations (red circles) are more likely to fall into different, potentially suboptimal, basins. (Adapted from Slide 20 of the presentation / Fig 4 of the paper)*

3.  **Functional Similarity:**
    LT solutions are not only close in weight space but also learn very similar functions to the pruned solution they originated from. This is measured by low "disagreement" (fraction of test images classified differently) between the LT solution and the pruned solution. Ensembles of LTs derived from the same pruning process show little performance gain, further suggesting they converge to nearly identical functions.

**The implication is powerful:** LTs aren't discovering new, highly effective sparse configurations through superior optimization dynamics. Instead, their specific initialization "nudges" the optimization process to rediscover a known good solution – the one found by pruning the dense network.

## Key Insights Summarized

This investigation into gradient flow in sparse neural networks reveals:

1.  **Sparsity-Aware Initialization Matters:** Naive use of dense initializations harms sparse networks by causing poor gradient flow from the start. Using initializations that account for the actual sparse connectivity is crucial.
2.  **Dynamic Sparse Training Boosts Gradient Flow:** DST methods improve gradient flow *during* training by adapting the network's sparse connections, leading to better generalization than training with fixed sparse masks.
3.  **Lottery Tickets are "Echoes" of Pruning:** LTs work well not because they inherently possess better gradient flow, but because their specific initial weights guide them to re-learn the solution of the pruned dense model they originated from. This limits their ability to find truly novel solutions in the sparse regime.

## Conclusion and Future Directions

Understanding gradient flow provides valuable insights into the challenges of training sparse neural networks. While sparsity-aware initializations and Dynamic Sparse Training offer promising avenues for improving how we train sparse models from scratch, the success of Lottery Tickets seems more about "remembering" a good solution than fundamentally solving the optimization difficulties in sparse landscapes.

The journey towards efficiently training sparse neural networks that are as performant as their dense counterparts, without relying on dense pre-training or specific "winning ticket" initializations, continues. Methods that can robustly navigate the complex loss landscapes of sparse models and maintain healthy gradient flow are key to unlocking the full potential of sparse AI.

## References
* **Evci2022GradientFlow**: Evci, U., Ioannou, Y., Keskin, C., & Dauphin, Y. (2022). Gradient Flow in Sparse Neural Networks and How Lottery Tickets Win. *arXiv preprint arXiv:2010.03533v2*. (Originally presented 2020/2021).
* **Frankle2019LTH**: Frankle, J., & Carbin, M. (2019). The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. In *International Conference on Learning Representations (ICLR)*.
* **Evci2020RigL**: Evci, U., Gale, T., Menick, J., Castro, P. S., & Elsen, E. (2020). Rigging the Lottery: Making All Tickets Winners. In *Proceedings of Machine Learning and Systems (MLSys)*.
* **Glorot2010Understanding**: Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. In *Proceedings of the thirteenth international conference on artificial intelligence and statistics (AISTATS)*.
* **He2015Delving**: He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. In *Proceedings of the IEEE international conference on computer vision (ICCV)*.
* **Liu2019Rethinking**: Liu, Z., Sun, M., Zhou, T., Huang, G., & Darrell, T. (2019). Rethinking the Value of Network Pruning. In *International Conference on Learning Representations (ICLR)*.

---

**Note on Figures:**
The `placeholder_figure_*.png` image paths above are illustrative. You will need to:
1.  Extract these figures from the provided PDF paper (e.g., "2010.03533v2.pdf") or the presentation (e.g., "Gradient Flow in Sparse Neural Networks (Vector Institute).pptx"). You might use screenshots or export images directly if the source allows.
2.  Save these images to a suitable path (e.g., an `assets/img/` directory relative to your Markdown file).
3.  Replace the placeholder paths in the Markdown with the correct paths to your images.
For example, if you save Figure 1 as `assets/img/training_outcomes.png`, you would change `![Training Outcomes](placeholder_figure_training_outcomes.png)` to `![Training Outcomes](assets/img/training_outcomes.png)`.
The figure captions are adapted from the paper and presentation.
