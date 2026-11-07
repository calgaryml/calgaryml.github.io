---
layout: page
permalink: /press/cerebras-reap-2025/
title: "REAP: One-Shot Pruning for Trillion-Parameter Mixture-of-Experts Models"
description: Archived copy of a Cerebras blog post
nav: false
sitemap: false
---

{% include press_archive_note.liquid
  description="a <a href='https://www.cerebras.ai/blog'>Cerebras blog</a> post"
  author="Mike Lasby, Ivan Lazarevich, Nish Sinnadurai, Sean Lie, Yani Ioannou and Vithursan Thangarasa"
  date="October 16, 2025"
  url="https://www.cerebras.ai/blog/reap"
%}

{% include figure.liquid path="assets/img/press/cerebras_reap/reap_overview.png" alt="Diagram of a sparse Mixture-of-Experts layer: a router network activates one of four experts for the input" class="img-fluid rounded z-depth-1" sizes="(min-width: 930px) 900px, 95vw" %}

**TL;DR:** We introduce REAP (Router-weighted Expert Activation Pruning), a new one-shot method for compressing Mixture-of-Experts (MoE) language models. Our key finding is that for generative tasks like code generation pruning low-impact experts is fundamentally better than merging them. REAP removes up to 50% of experts from models as large as 1 trillion parameters while largely maintaining baseline model quality. For instance, with the Qwen3-480B-Coder-FP8 model, REAP at 50% pruning retains 97.6% of its baseline non-agentic coding ability and 96.7% on the agentic SWE-Bench benchmark. We are open-sourcing the complete codebase and pruned model checkpoints on HuggingFace to encourage further research.

### Leveraging Expert Redundancy for MoE Compression

Sparsely-activated Mixture-of-Experts (SMoE) models achieve their high quality by **decoupling their total parameter count from their computational cost** [1]. This allows them to leverage a much larger parameter budget for greater model capacity, while only activating a small, computationally efficient subset of “expert” networks for any given input. However, this efficiency comes with a steep memory cost. Models such as Qwen3-480B-Coder and Kimi-K2 contain hundreds of billions of parameters, even though only a fraction are active at any time. This memory footprint is a significant barrier to deployment and research.

Prior works show a critical insight: large MoE models contain significant **expert redundancy** [2, 3]. Not all experts contribute equally to the tasks we care about, and expert usage is often highly imbalanced. Some experts are rarely chosen by the model's router, and their impact on the final output can be minimal even when they are selected. This redundancy creates an opportunity for model compression. By identifying and removing these low-impact experts, we can achieve a significant reduction in memory footprint without degrading model quality. This leads to a fundamental question: is it better to remove these experts entirely with pruning or to combine them with others with merging? While merging seems intuitively appealing, our work reveals a critical flaw in that approach.

### What Goes Wrong with Expert Merging

Recent work suggests that merging experts is superior to pruning them [4, 5]. These results seem intuitive; instead of discarding an expert entirely, why not average its weights with a similar one to preserve some of its learned information? Early studies confirm this, showing that merging outperforms pruning on discriminative tasks like multiple-choice question answering.

However, our work shows this **does not hold true for generative tasks**. In our experiments, pruning proves to be the superior strategy, consistently achieving higher model quality than merging across all generative benchmarks.These are tasks like code generation, tool calling, mathematical reasoning, or creative writing, which require the model to produce diverse or structured output instead of just picking an answer from a list. This finding highlights that compression strategies optimized for discriminative tasks may not directly translate to generative settings.

### Expert Merging Challenges: Functional Subspace Collapse

Our work shows that merging experts introduces a fundamental and **irreducible error**. To understand why, we first need to look at how a healthy MoE layer works. The router's main job is to perform **input-dependent mixing**. For example, it might combine 70% of Expert A with 30% of Expert B for one token, then adapt to a 40/60 mix for the next. This dynamic ability is crucial for generating high-quality, nuanced output.

Current merging techniques inhibit this capability as they combine the router’s gate-values for the merged experts by summation. When you merge experts A and B, you replace them with a single, static average. The router loses its freedom to choose; it is now forced to use that one fixed average for *all* inputs. This loss of dynamic control is what causes the irreducible error. We call this outcome **functional subspace collapse**, because the range of possible outputs the model can produce dramatically shrinks. You simply cannot recreate the flexibility of dynamic mixing from a static average.

**The Core Intuition -** The irreducible error from merging is proportional to:

- How much the router varies its mixing strategy (policy variability)
- How different the two experts are (expert gap)
- The magnitude of their gate-values (router scale)

Pruning avoids this error by maintaining the router's operational freedom. When an expert is removed, the router's control over all *surviving* experts remains completely independent. As a result, the router can continue to modulate each one dynamically, fully preserving the model's ability to adapt its mixing strategy based on the input.

{% include figure.liquid path="assets/img/press/cerebras_reap/fig1_layer0.png" alt="Expert activations of Qwen3-30B-A3B layer 0 projected onto two principal components, for the original, REAP-pruned and merged models" caption="Figure 1 Qwen3-30B-A3B Layer 0. In this early, more general layer, the collapse from merging is visible but less severe, validating our theory that the error is worse when specialized experts with high policy variability are merged." class="img-fluid rounded z-depth-1" sizes="(min-width: 930px) 900px, 95vw" %}

{% include figure.liquid path="assets/img/press/cerebras_reap/fig2_layer47.png" alt="Expert activations of Qwen3-30B-A3B layer 47 projected onto two principal components, for the original, REAP-pruned and merged models" caption="Figure 2 Qwen3-30B-A3B Layer 47. In this late, specialized layer, the original experts (left) show wide functional diversity. After pruning 50% of experts with REAP (center), the surviving experts (blue) perfectly maintain the original distribution's shape and breadth. However, merging 50% of the experts (right) causes a catastrophic collapse. All merged experts are compressed into a tiny, functionally limited cluster, representing a nearly 100x reduction in diversity." class="img-fluid rounded z-depth-1" sizes="(min-width: 930px) 900px, 95vw" %}

We visualized this collapse by projecting expert activations onto their first two principal components across different layers of a model. For example, in early layers (Layer 0 shown in Figure 1), the original 128 experts of Qwen3-30B-A3B form a compact distribution along the diagonal. After pruning 50% of the experts, the 64 surviving experts maintain the same geometric structure, overlaying the original distribution faithfully. Merging, however, contracts the distribution of expert activations toward the center, a visible but modest compression.

The contrast becomes more evident in late layers (Layer 47 shown in Figure 2), where experts have specialized for distinct computational roles. The original experts span a wide range from PC1 coordinates of approximately -100 to 200. Pruning preserves this full breadth with 64 experts distributed across the same space. But merging induces catastrophic collapse: all merged experts compress into a tight cluster near the center, representing nearly a 100x reduction in functional diversity. This dramatic difference validates our theoretical analysis that irreducible error is proportional to policy variability. Early layers exhibit lower policy variability and thus modest collapse, while late layers with high policy variability suffer severe functional collapse when specialized experts are merged.

### REAP: Pruning by Impact, Not Just Frequency

If pruning is the right path, how do we decide which experts to remove? Simple frequency counts are not enough because they ignore an expert's actual contribution. REAP provides a more robust way to measure an expert's true importance. It effectively asks two simple questions:

1. How often and how strongly does the model's router **choose** this expert? (This is measured by the gate-value).
1. When chosen, how much does the expert actually **change** the final result? (This is measured by its output magnitude).

By combining these two factors, REAP identifies experts that are both rarely used and have little impact when they are. This allows us to prune the least important experts while preserving the model's core capabilities.

REAP directly measures an expert's contribution to the output of its MoE layer. Specifically, the saliency score, $$S_j$$, is defined as the average of this contribution over tokens for which the expert is active:

$$
S_j = \frac{1}{|X_j|} \sum_{x \in X_j} g_j(x) \cdot \| f_j(x) \|_2,
$$

where $$f_j(x)$$ is the output of expert $$j$$, $$g_j(x)$$ is the router's gate-value for expert $$j$$ on input $$x$$, and $$X_j$$ is the set of inputs where $$( g_j(x) \in \text{TopK}(\mathbf{g}(x))$$. Put simply, this means we only average over the inputs where the router actually activated expert $$j$$ as one of its top choices. This score thus captures an expert's true importance by multiplying its selection weight (the gate-value) by its functional impact (the output magnitude), giving us a direct measure of what to preserve.

### Putting It to the Test: REAP vs. Baselines

We evaluated REAP against well-established expert merging methods (M-SMoE [4], HC-SMoE [5]) and other pruning approaches across 7 diverse MoE models (21B to 1T parameters). EvalPlus represents the average accuracy across HumanEval, HumanEval+, MBPP, and MBPP+ benchmarks using the [EvalPlus](https://github.com/evalplus/evalplus) [6, 7] framework, while LiveCodeBench evaluation was conducted using [EvalScope](https://github.com/modelscope/evalscope) [8]. For SWE-Bench evaluation, we ran our compressed models with the mini-SWE-agent scaffolding [9] and report the score on the SWE-Bench Verified test set [10]. For BFCL-v3, we use the original Gorilla [11] framework for evaluating our models. All evaluations on generative benchmarks used greedy decoding.

**REAP outperforms merging at both compression ratios.** On Qwen3-30B-A3B and GLM-4.5-Air, REAP maintains near-baseline quality even at 25% compression (within 1 percentage point), while merging methods degrade by 2-5 points (see Table 1). At 50% compression, merging suffers a severe drop, while REAP retains high model quality.

{% include figure.liquid path="assets/img/press/cerebras_reap/table1.png" alt="Table 1: REAP vs. M-SMoE and HC-SMoE merging on Qwen3-30B-A3B and GLM-4.5-Air at 25% and 50% compression" class="img-fluid rounded z-depth-1" sizes="(min-width: 930px) 900px, 95vw" %}

The impact of 50% compression highlights the fundamental difference between the methods. For Qwen3-30B-A3B, REAP's model quality shows a small degradation, retaining 95.9% of the baseline's code generation capabilities. In contrast, the merging method HC-SMoE sees its quality falling to just 65.2% of the baseline. We observe the same pattern on GLM-4.5-Air, where REAP preserves 94.1% of the baseline quality on LiveCodeBench, while HC-SMoE drops to 58.8%. This consistent gap demonstrates that merging is a far more destructive operation than REAP at high compression ratios.

### REAP scales to Trillion-Parameter MoE models

For Qwen3-Coder-480B-FP8 and Kimi-K2-Instruct-W4A16 (1T parameters), REAP maintains near-lossless quality at 50% compression (refer to Table 2). This demonstrates that REAP is not just a standalone technique. It can be applied on top of other memory-saving methods like quantization to achieve even greater compression, making it a valuable tool for practical deployment.

{% include figure.liquid path="assets/img/press/cerebras_reap/table2.png" alt="Table 2: REAP vs. frequency-based pruning and EAN on Qwen3-480B-Coder-FP8 and Kimi-K2-Instruct-W4A16" class="img-fluid rounded z-depth-1" sizes="(min-width: 930px) 900px, 95vw" %}

The results highlight REAP's robustness not just on standard benchmarks, but across a spectrum of task modalities. At a high compression factor of 50% on the Qwen3-480B-Coder-FP8 model, REAP exhibits a noticeably uniform preservation of model quality, **retaining over 96% of the baseline's capabilities across non-agentic coding**, **agentic coding**, and **tool-use** tasks. This provides a clear advantage over EAN [12], which shows more variability and greater degradation on key generative tasks. This high-fidelity preservation stands in contrast to naive heuristics like frequency-based pruning, where the model loses its generative ability to produce coherent output. These results demonstrate the efficacy of our proposed saliency criterion for holistically preserving a model's capabilities during pruning.

### Exploring the Frontier of MoE Compression

Our findings open several avenues for future research in MoE compression. Further exploration into the dynamics of router policy variability and expert specialization during training could yield deeper insights into which experts become redundant and why. Investigating the combination of REAP with other compression techniques such as quantization, low-rank decomposition, and iterative pruning schedules could unlock even greater efficiency gains. The provided code and compressed model checkpoints aim to facilitate such continued investigation within the community. Ultimately, a more comprehensive understanding of expert redundancy and router coordination holds the potential to unlock new methodologies for making frontier MoE models more practical and efficient. By demonstrating that preserving the router's independent control over experts is fundamental to maintaining model quality, we hope REAP inspires new approaches to compression that respect the architectural principles underlying these powerful models.

### Resources

**Paper**:

[REAP the Experts: Why Pruning Prevails for One-Shot MoE compression](https://arxiv.org/abs/2510.13999)

**Code**: [https://github.com/CerebrasResearch/reap](https://github.com/CerebrasResearch/reap)

**HuggingFace Models**:

[Qwen3-Coder-REAP-246B-A35B-FP8](https://huggingface.co/cerebras/Qwen3-Coder-REAP-246B-A35B-FP8)

[Qwen3-Coder-REAP-363B-A35B-FP8](https://huggingface.co/cerebras/Qwen3-Coder-REAP-363B-A35B-FP8)

### Authors

Mike Lasby, Ivan Lazarevich, Nish Sinnadurai, Sean Lie, Yani Ioannou, Vithursan Thangarasa

### References

[1] Abnar et al., “Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models”, _arXiv_ (2025).

[2] Chen et al., “Task-Specific Expert Pruning for Sparse Mixture-of-Experts”, _arXiv_ (2022).

[3] Lu et al., “Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models”, _In ACL_ (2024).

[4]Li et al., “Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy”, _In ICLR_ (2024).

[5] Chen et al., “Retraining-Free Merging of Sparse MoE via Hierarchical Clustering”, _In ICML_ (2025).

[6] Liu et al., “Is Your Code Generated by Chat{GPT} Really Correct? Rigorous Evaluation of Large Language Models for Code Generation”, _In NeurIPS_ (2023).

[7] Liu et al., “Evaluating Language Models for Efficient Code Generation”, _In COLM_ (2024).

[8] [https://github.com/modelscope/evalscope](https://github.com/modelscope/evalscope)

[9] Yang et al., “SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering”, _In NeurIPS_ (2025).

[10] Chowdhury et al., “Introducing SWE-bench Verified”, [https://openai.com/index/introducing-swe-bench-verified/](https://openai.com/index/introducing-swe-bench-verified/) (2025).

[11] Patil et al., “Gorilla: Large Language Model Connected with Massive APIs”, _In NeurIPS_ (2024).

[12] Jaiswal et al., “Finding Fantastic Experts in MoEs: A Unified Study for Expert Dropping Strategies and Observations”, _arXiv_ (2025).
