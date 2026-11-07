---
layout: page
permalink: /press/deleting-weights-2026/
title: "The Surprising Effectiveness of Deleting Weights in LLM Post-Training"
description: Archived copy of the Agentic Learning AI Lab project page
nav: false
sitemap: false
---

{% include press_archive_note.liquid
  description="the <a href='https://agenticlearning.ai'>Agentic Learning AI Lab</a> project page"
  author="Jack Lu, Zhenbang Yang, Mike Lasby, Tejas Pote, Yani Ioannou and Mengye Ren"
  archived="November 7, 2026"
  url="https://agenticlearning.ai/deleting-weights/"
%}

Jack Lu<sup>1\*</sup>, Zhenbang Yang<sup>1\*</sup>, Mike Lasby<sup>2</sup>, Tejas Pote<sup>2</sup>, Yani Ioannou<sup>2</sup> and Mengye Ren<sup>1</sup><br>
<sup>1</sup>New York University, <sup>2</sup>University of Calgary, <sup>\*</sup>Equal contribution

_Advances in Neural Information Processing Systems 39 (NeurIPS 2026)_

[PDF](https://openreview.net/pdf?id=9Z9V0zY5As) · [Poster](/assets/pdf/deleting_weights_poster_neurips2026.pdf) · [Code](https://github.com/agentic-learning-ai-lab/deleting-weights)

**TL;DR:** Setting fewer than 0.05% of a pretrained LLM's weights to zero, with no other change, matches full fine-tuning at scale across reinforcement learning, supervised fine-tuning, and on-policy distillation.

## Abstract

How little of a pretrained large language model (LLM) has to change for it to acquire new capabilities? We find that zeroing fewer than 0.05% of its weights, with no other modification, matches full fine-tuning (FFT) at scale across three LLM post-training regimes: reinforcement learning with verifiable rewards, supervised fine-tuning, and on-policy distillation. Our method, Deletion Tuning, learns a binary mask over the gate projection of transformer blocks. Two practical advantages follow: the trained mask is hundreds to thousands of times smaller than the corresponding FFT checkpoint, and Deletion Tuning forgets far less, which addresses a central challenge in continual learning. Under reinforcement learning, its prior-task accuracy stays near the base level while FFT, LoRA, and random sparse fine-tuning lose up to 5.4 points. To understand why masking alone suffices, we analyze the geometry of the learned weight changes under reinforcement learning using stable rank, spectrum drift, and principal-weight overlap. Compared with LoRA and random sparse fine-tuning, Deletion Tuning is closest to FFT across the three measures.

## Overview

<figure>
<video class="img-fluid rounded z-depth-1" src="{{ 'assets/video/press/deleting_weights_overview.mp4' | relative_url }}" poster="{{ 'assets/img/press/deleting_weights/overview_poster.webp' | relative_url }}" autoplay muted loop playsinline controls aria-label="Animation of Deletion Tuning: the logits G change during training while the sigmoid mask sharpens into a step, and the deployed weight matrix crosses out the weights whose logits are negative."></video>
<figcaption class="caption"><strong>Deletion Tuning learns which pretrained weights to delete and writes no new values.</strong> The pretrained weights \(W\) stay frozen, and each is paired with a trainable logit in \(G\), the only parameters that post-training updates. During training, each weight is multiplied by \(\sigma(\alpha G)\), a smooth mask between 0&nbsp;and&nbsp;1. The sharpness \(\alpha\) rises over training until the sigmoid becomes a hard threshold at \(G = 0\). At deployment, weights with positive logits (blue) keep their pretrained values and weights with negative logits (red) are set to zero (crossed out), so the trained adapter is one bit per weight.</figcaption>
</figure>

Full fine-tuning (FFT), LoRA, and sparse fine-tuning all write new values into a pretrained model. We introduce _Deletion Tuning_ to test whether deleting pretrained weights alone can match FFT.

- **Deletion alone matches full fine-tuning.** Deletion Tuning is on par with FFT at the largest scale we test in [GRPO](https://arxiv.org/abs/2402.03300), SFT, and on-policy distillation ([SDFT](https://arxiv.org/abs/2601.19897)), up to a 30B mixture-of-experts model. Under GRPO, it also solves problems that the base model fails in all 128 sampled attempts.
- **Deletion Tuning is compact and forgets less.** Its masks are orders of magnitude smaller than FFT checkpoints in all three regimes, and it keeps prior-task accuracy throughout training.
- **Deletion Tuning's learned updates have FFT-like geometry.** Its weight updates resemble FFT's on three geometric measures, which helps explain its accuracy.

## Deletion Tuning

Deletion Tuning freezes the pretrained model and gives every weight of each gate projection its own trainable logit. During training, the weight matrix $$W$$ is multiplied elementwise by the soft mask $$\sigma(\alpha G)$$, an annealed sigmoid of the logits $$G$$ that lets gradients reach them:

$$
\widetilde{W} = W \odot \sigma(\alpha G).
$$

The sharpness $$\alpha$$ rises linearly over training, so the soft mask approaches the hard mask that Deletion Tuning deploys, a binary mask that keeps each pretrained weight with a positive logit and deletes the rest:

$$
\widehat{W} = W \odot \mathbf{1}[G > 0].
$$

We initialize the logits from [Wanda](https://arxiv.org/abs/2306.11695) pruning scores and set no sparsity target, so training decides how many weights to delete. Bit-packed and compressed with [zstd](https://www.rfc-editor.org/rfc/rfc8878), the hard mask takes only 1.0 MB for Llama1B and 8.4 MB for Llama8B, whose fully fine-tuned checkpoint takes 30,633 MB. This makes adapters cheap to store and to switch: changing tasks only loads a small bitmask and scatters zeros into the shared base weights, rather than loading a separate full model checkpoint.

## Matching Full Fine-Tuning with Reinforcement Learning

We train Deletion Tuning with [GRPO](https://arxiv.org/abs/2402.03300) in two settings. The first trains [Llama3](https://arxiv.org/abs/2407.21783) Instruct models from 1B to 8B on [GSM8K](https://arxiv.org/abs/2110.14168). The second trains [Qwen2.5](https://arxiv.org/abs/2412.15115) base models from 0.5B to 14B and the [Qwen3-30B-A3B](https://arxiv.org/abs/2505.09388) mixture-of-experts model on the [SimpleRL-Zoo](https://arxiv.org/abs/2503.18892) math corpus and evaluates them on six held-out math benchmarks. We name each model by its family and size, such as Llama8B and Qwen7B. Besides FFT, we compare against [LoRA](https://arxiv.org/abs/2106.09685) at ranks 1 and 4 and against random sparse fine-tuning (MTH, from [The Multiple Ticket Hypothesis](https://arxiv.org/abs/2602.01599)) at 99% and 99.95% sparsity, across six backbones from 0.5B to 8B. Every Deletion Tuning run deletes fewer than 0.05% of the model's weights.

<figure class="press-table">
<figcaption><strong>Deletion Tuning has the smallest adapter at every scale and matches FFT at the largest backbone of each family.</strong> Best-checkpoint validation accuracy (%) and adapter size (MB) under GRPO, on GSM8K for Llama and on the six-benchmark SimpleRL-Zoo mean for Qwen.</figcaption>
<div class="table-responsive">
<table class="table table-sm">
<thead>
<tr><th></th><th colspan="6">Llama, GSM8K</th><th colspan="6">Qwen, SimpleRL-Zoo</th></tr>
<tr><th></th><th colspan="2">1B</th><th colspan="2">3B</th><th colspan="2">8B</th><th colspan="2">0.5B</th><th colspan="2">3B</th><th colspan="2">7B</th></tr>
<tr><th>Method</th><th>Acc.</th><th>Size</th><th>Acc.</th><th>Size</th><th>Acc.</th><th>Size</th><th>Acc.</th><th>Size</th><th>Acc.</th><th>Size</th><th>Acc.</th><th>Size</th></tr>
</thead>
<tbody>
<tr><td>Base</td><td>1.21</td><td>0</td><td>53.53</td><td>0</td><td>60.27</td><td>0</td><td>10.44</td><td>0</td><td>37.73</td><td>0</td><td>40.18</td><td>0</td></tr>
<tr><td>FFT</td><td>60.50</td><td>4714</td><td><strong>82.49</strong></td><td>12256</td><td>83.24</td><td>30633</td><td>19.01</td><td>1885</td><td><strong>44.35</strong></td><td>11772</td><td>51.89</td><td>29051</td></tr>
<tr><td>LoRA r=1</td><td>59.29</td><td>2.7</td><td>82.39</td><td>5.8</td><td>83.38</td><td>10.0</td><td>20.30</td><td>2.1</td><td>43.55</td><td>7.1</td><td>52.13</td><td>9.6</td></tr>
<tr><td>LoRA r=4</td><td>60.58</td><td>10.8</td><td>82.44</td><td>23.2</td><td>82.87</td><td>40.0</td><td>20.30</td><td>8.4</td><td>43.59</td><td>28.6</td><td>51.70</td><td>38.5</td></tr>
<tr><td>MTH 99%</td><td><strong>60.73</strong></td><td>68.2</td><td>80.53</td><td>177.4</td><td>83.31</td><td>443.6</td><td><strong>21.84</strong></td><td>27.3</td><td>44.16</td><td>170.4</td><td>52.03</td><td>420.7</td></tr>
<tr><td>MTH 99.95%</td><td>58.61</td><td>4.1</td><td>82.17</td><td>10.7</td><td>83.07</td><td>26.6</td><td>20.16</td><td>1.7</td><td>43.87</td><td>13.7</td><td>51.24</td><td>25.3</td></tr>
<tr><td>Deletion Tuning</td><td>54.44</td><td><strong>1.0</strong></td><td>81.49</td><td><strong>3.1</strong></td><td><strong>83.55</strong></td><td><strong>8.4</strong></td><td>17.64</td><td><strong>0.5</strong></td><td>41.98</td><td><strong>2.7</strong></td><td><strong>52.32</strong></td><td><strong>6.9</strong></td></tr>
</tbody>
</table>
</div>
</figure>

Tuning FFT's learning rate and KL coefficient does not change the parity at 7B and 8B. At smaller sizes, Deletion Tuning recovers 90% to 99% of FFT's accuracy, and the gap closes as models grow. For the mixture-of-experts model, we mask the gate projection inside each of its 128 experts and leave the router unchanged.

<figure class="press-table">
<figcaption><strong>Deletion Tuning matches FFT on dense and mixture-of-experts models from 7B to 30B with an adapter thousands of times smaller.</strong> Best-checkpoint validation accuracy (%) and adapter size (MB) under GRPO.</figcaption>
<div class="table-responsive">
<table class="table table-sm">
<thead>
<tr><th></th><th colspan="2">Llama8B</th><th colspan="2">Qwen7B</th><th colspan="2">Qwen14B</th><th colspan="2">Qwen3-30B-A3B</th></tr>
<tr><th></th><th colspan="2">GSM8K</th><th colspan="2">SimpleRL-Zoo</th><th colspan="2">SimpleRL-Zoo</th><th colspan="2">SimpleRL-Zoo</th></tr>
<tr><th>Method</th><th>Acc.</th><th>Size</th><th>Acc.</th><th>Size</th><th>Acc.</th><th>Size</th><th>Acc.</th><th>Size</th></tr>
</thead>
<tbody>
<tr><td>Base</td><td>60.27</td><td>0</td><td>40.18</td><td>0</td><td>44.37</td><td>0</td><td>49.41</td><td>0</td></tr>
<tr><td>FFT</td><td>83.24</td><td>30633</td><td>51.89</td><td>29051</td><td>54.87</td><td>56343</td><td>58.43</td><td>116471</td></tr>
<tr><td>Deletion Tuning</td><td><strong>83.55</strong></td><td><strong>8.4</strong></td><td><strong>52.32</strong></td><td><strong>6.9</strong></td><td><strong>55.43</strong></td><td><strong>11.4</strong></td><td><strong>58.54</strong></td><td><strong>29.6</strong></td></tr>
</tbody>
</table>
</div>
</figure>

### Acquiring New Capabilities

On pass@k, Deletion Tuning leads the base model at every $$k$$. Even at $$k = 128$$, the base model stays 4.3 and 6.1 points below Deletion Tuning on Llama8B and Qwen7B.

<figure class="press-table">
<figcaption><strong>Deletion Tuning acquires new capabilities.</strong> Pass@k (%), the share of problems solved within \(k\) attempts, estimated from 128 samples per problem at temperature 1.0. A lead at large \(k\) means solving problems the base model misses even with many attempts.</figcaption>
<div class="table-responsive">
<table class="table table-sm">
<thead>
<tr><th></th><th colspan="5">Llama8B, GSM8K</th><th colspan="5">Qwen7B, SimpleRL-Zoo</th></tr>
<tr><th>Method</th><th><em>k</em>=1</th><th>2</th><th>8</th><th>32</th><th>128</th><th><em>k</em>=1</th><th>2</th><th>8</th><th>32</th><th>128</th></tr>
</thead>
<tbody>
<tr><td>Base</td><td>53.20</td><td>66.75</td><td>83.64</td><td>92.40</td><td>95.29</td><td>27.35</td><td>38.75</td><td>56.18</td><td>66.26</td><td>74.01</td></tr>
<tr><td>Deletion Tuning</td><td><strong>81.75</strong></td><td><strong>88.51</strong></td><td><strong>95.75</strong></td><td><strong>98.44</strong></td><td><strong>99.62</strong></td><td><strong>49.39</strong></td><td><strong>53.22</strong></td><td><strong>64.64</strong></td><td><strong>73.23</strong></td><td><strong>80.08</strong></td></tr>
</tbody>
</table>
</div>
</figure>

### Learning the Mask Drives the Gains

Wanda one-shot pruning supplies Deletion Tuning's starting scores. Applied without training, it changes accuracy by at most 2.4 points on all six backbones, while Deletion Tuning gains up to 53.2 points.

<figure class="press-table">
<figcaption><strong>Wanda alone stays near the base model.</strong> Validation accuracy (%) of the base model, of Wanda one-shot pruning at the best of eight sparsity levels, and of the best checkpoint of the Deletion Tuning mask trained with GRPO.</figcaption>
<div class="table-responsive">
<table class="table table-sm">
<thead>
<tr><th></th><th colspan="3">Llama, GSM8K</th><th colspan="3">Qwen, SimpleRL-Zoo</th></tr>
<tr><th>Method</th><th>1B</th><th>3B</th><th>8B</th><th>0.5B</th><th>3B</th><th>7B</th></tr>
</thead>
<tbody>
<tr><td>Base</td><td>1.21</td><td>53.53</td><td>60.27</td><td>10.44</td><td>37.73</td><td>40.18</td></tr>
<tr><td>Wanda (best \(\eta\))</td><td>1.59</td><td>52.46</td><td>60.65</td><td>11.42</td><td>38.70</td><td>42.49</td></tr>
<tr><td>Deletion Tuning</td><td><strong>54.44</strong></td><td><strong>81.49</strong></td><td><strong>83.55</strong></td><td><strong>17.64</strong></td><td><strong>41.98</strong></td><td><strong>52.32</strong></td></tr>
</tbody>
</table>
</div>
</figure>

## Matching Full Fine-Tuning beyond Reinforcement Learning

Deletion Tuning even matches FFT under SFT on [MMLU-Pro](https://arxiv.org/abs/2406.01574) and [BBEH](https://arxiv.org/abs/2502.19187), and under the on-policy distillation method [Self-Distillation Fine-Tuning](https://arxiv.org/abs/2601.19897) (SDFT) on tool use ([ToolAlpaca](https://arxiv.org/abs/2306.05301)) and science questions ([Science](https://arxiv.org/abs/2406.09098)). SFT imitates fixed targets token by token and [produces dense updates](https://arxiv.org/abs/2505.11711), while reinforcement-learning updates are sparse. In SDFT, the student samples its own responses and a teacher supplies a target at every token.

<figure class="press-table">
<figcaption><strong>Under SFT, Deletion Tuning matches FFT at 3B.</strong> Best-checkpoint evaluation accuracy (%), averaged over the 14 MMLU-Pro subjects and the 23 BBEH subtasks, and adapter size (MB) for Llama1B and Llama3B.</figcaption>
<div class="table-responsive">
<table class="table table-sm">
<thead>
<tr><th></th><th colspan="4">MMLU-Pro</th><th colspan="4">BBEH</th></tr>
<tr><th></th><th colspan="2">1B</th><th colspan="2">3B</th><th colspan="2">1B</th><th colspan="2">3B</th></tr>
<tr><th>Method</th><th>Acc.</th><th>Size</th><th>Acc.</th><th>Size</th><th>Acc.</th><th>Size</th><th>Acc.</th><th>Size</th></tr>
</thead>
<tbody>
<tr><td>Base</td><td>10.37</td><td>0</td><td>12.63</td><td>0</td><td>2.17</td><td>0</td><td>8.87</td><td>0</td></tr>
<tr><td>FFT</td><td>16.97</td><td>4714</td><td>21.19</td><td>12256</td><td><strong>8.35</strong></td><td>4714</td><td><strong>11.39</strong></td><td>12256</td></tr>
<tr><td>Deletion Tuning</td><td><strong>18.62</strong></td><td><strong>1.0</strong></td><td><strong>22.28</strong></td><td><strong>3.1</strong></td><td>5.65</td><td><strong>1.0</strong></td><td><strong>11.39</strong></td><td><strong>3.1</strong></td></tr>
</tbody>
</table>
</div>
</figure>

<figure class="press-table">
<figcaption><strong>Under SDFT, Deletion Tuning matches FFT on Qwen7B.</strong> Best-checkpoint exact-match accuracy (%) and adapter size (MB) on ToolAlpaca tool use and Science.</figcaption>
<div class="table-responsive">
<table class="table table-sm">
<thead>
<tr><th></th><th colspan="2">ToolAlpaca</th><th colspan="2">Science</th></tr>
<tr><th>Method</th><th>Acc.</th><th>Size</th><th>Acc.</th><th>Size</th></tr>
</thead>
<tbody>
<tr><td>Base</td><td>41.24</td><td>0</td><td>32.35</td><td>0</td></tr>
<tr><td>FFT</td><td>68.04</td><td>29051</td><td><strong>71.20</strong></td><td>29051</td></tr>
<tr><td>Deletion Tuning</td><td><strong>69.07</strong></td><td><strong>27.1</strong></td><td>70.61</td><td><strong>47.1</strong></td></tr>
</tbody>
</table>
</div>
</figure>

## Deletion Tuning Forgets Less

During GRPO on GSM8K, Deletion Tuning's mean accuracy on six prior tasks ([HellaSwag](https://arxiv.org/abs/1905.07830), [MMLU](https://arxiv.org/abs/2009.03300), [TruthfulQA](https://arxiv.org/abs/2109.07958), [WinoGrande](https://arxiv.org/abs/1907.10641), [IFEval](https://arxiv.org/abs/2311.07911), and [HumanEval](https://arxiv.org/abs/2107.03374)) stays within one point of the base model. FFT ends 1.3 to 3.1 points below the base, and LoRA and MTH fall up to 5.4 points. Following [RL's Razor](https://arxiv.org/abs/2509.04259), we also track the forward KL to the base policy: Deletion Tuning's stays at or below 0.05, while FFT's reaches 1.0 on Llama1B and 1.5 on Llama8B. MTH at 99.95% sparsity trains 0.05% of all parameters, comparable to the 0.035% to 0.046% that Deletion Tuning deletes at these scales, which suggests that deletion itself keeps forgetting low. Learning new tasks while keeping old ones makes Deletion Tuning a promising candidate for continual learning.

<figure>
<img class="img-fluid rounded z-depth-1" src="{{ 'assets/img/press/deleting_weights/retention.png' | relative_url }}" alt="Forward KL and change in prior-task accuracy over GRPO training on Llama1B, Llama3B, and Llama8B: Deletion Tuning stays near zero on both while FFT, LoRA, and MTH drift." loading="lazy">
<figcaption class="caption"><strong>Deletion Tuning keeps prior-task accuracy at every Llama scale while the baselines drift.</strong> (a) Forward KL between the trained and base policies on GSM8K during GRPO training, the forgetting predictor of RL's Razor. (b) Change in mean accuracy on a six-task prior-task suite from the base model (dashed line).</figcaption>
</figure>

## Why Deletion Tuning Works

We compare each method's weight updates on three measures of update geometry: stable rank (how many directions an update uses), spectrum drift (how far the singular values move), and principal-weight overlap (how many changed weights fall on the base model's principal weights). The last two come from [The Path Not Taken](https://arxiv.org/abs/2511.08567), which studies the updates of reinforcement learning with verifiable rewards. Like FFT, Deletion Tuning barely moves the spectrum (its drift stays within 1.4× of FFT's) and changes fewer principal weights than chance (an overlap of 0.43 to 0.46, against FFT's 0.41 to 0.46), while LoRA stays near the random baseline of 0.5 and MTH sits at it.

<figure>
<img class="img-fluid rounded z-depth-1" src="{{ 'assets/img/press/deleting_weights/geometry.png' | relative_url }}" alt="Stable rank, spectrum drift, and principal-weight overlap of each method's update for Llama8B and Qwen7B: Deletion Tuning is the method closest to FFT on all three." loading="lazy">
<figcaption class="caption"><strong>Deletion Tuning (blue) tracks FFT (red) most closely across three update-geometry metrics.</strong> Values at the saved GRPO checkpoint for Llama8B (filled) and Qwen7B (open): (a) stable rank of \(\Delta W\), (b) spectrum drift, and (c) principal-weight overlap.</figcaption>
</figure>

LoRA caps the rank of its update, and MTH trains weights chosen at random before training. Deletion Tuning has neither restriction, which helps explain how its updates come close to FFT's without writing new values.

## Deletion Tuning Is Robust to Hyperparameter Changes

Deletion Tuning works across a range of values for its hyperparameters: the initial sparsity (how many weights the Wanda initialization deletes), the learning rate, and the regularization strength (how strongly the mask is pushed to keep large weights). Across all 12 combinations we sweep, every setting lands within 1.6 points of the best run on Llama8B GSM8K and within 3.2 points on Qwen7B SimpleRL-Zoo, and varying the initial sparsity fourfold changes accuracy by at most 0.8 and 1.7 points.
