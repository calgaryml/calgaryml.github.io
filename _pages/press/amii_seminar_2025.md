---
layout: page
permalink: /press/amii-seminar-2025/
title: "Amii AI Seminar: Sparse Training from Random Initialization: Aligning Lottery Ticket Masks using Weight Symmetry"
description: Archived excerpt from the Amii AI Seminar Series 2025 listing
nav: false
sitemap: false
---

{% include press_archive_note.liquid
  excerpt=true
  description="the <a href='https://sites.google.com/ualberta.ca/ai-seminar'>Amii AI Seminar Series</a> 2025 listing, University of Alberta"
  date="November 21, 2025"
  url="https://sites.google.com/ualberta.ca/ai-seminar/past-seminars/2025"
%}

### November 21, 2025

{% include figure.liquid path="assets/img/press/amii_seminar_2025.jpg" alt="Yani Ioannou presenting to a seminar audience" class="img-fluid rounded z-depth-1" sizes="(min-width: 930px) 900px, 95vw" %}

**Speaker:** Dr. Yani Ioannou, Assistant Professor and Schulich Research Chair from the University of Calgary, hosted by Dr. Martha White

**Title:** Sparse Training from Random Initialization: Aligning Lottery Ticket Masks using Weight Symmetry

**Abstract:** The Lottery Ticket Hypothesis (LTH) suggests there exists a sparse LTH mask and weights that achieve the same generalization performance as the dense model while using significantly fewer parameters. However, finding a LTH solution is computationally expensive, and the LTH sparsity mask does not generalize to other random weight initializations. Recent work has suggested that neural networks trained from random initialization find solutions within the same basin modulo permutation and proposes a method to align trained models within the same loss basin. We hypothesize that misalignment of basins is the reason why sparse masks do not generalize to new random initializations and propose permuting the LTH mask to align with the new optimization basin when performing sparse training from a different random init. We empirically show a significant increase in generalization when sparse training from random initialization with the permuted mask as compared to using the non-permuted LTH mask, on multiple datasets and models.

**Presenter bio:** Yani Ioannou is an Assistant Professor and Schulich Research Chair in the Department of Electrical and Software Engineering of the Schulich School of Engineering, at the University of Calgary in Canada, Alberta. Yani was previously a Visiting Researcher at Google Brain Toronto (DeepMind) with Prof. Geoffrey Hinton, and a Post-doctoral Fellow at the Vector Institute with Dr. Graham Taylor. Yani completed his PhD at the University of Cambridge in 2018 supported by a Microsoft Research Ph.D. Scholarship, where he was supervised by Prof. Roberto Cipolla and Dr. Antonio Criminisi.

**Website:** [The Calgary Machine Learning Lab](https://www.calgaryml.com/)

**Slides:** [20251121 AI Seminar Slides, Sparse Rebasin, Yani Ioannou](https://drive.google.com/file/d/1h1Lbtvz8unyXrIZw9hYnFYz6pHbrkr_6/view)

**Video:** [Watch on YouTube](https://youtu.be/WW1ksk-O5c0)
