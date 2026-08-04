---
layout: about
title: about
permalink: /
subtitle:
nav_order: 0

profile:
  align: left
  image: ucmllogo-text.svg
  name: Calgary ML Lab
  image_circular: false # crops the image to make it circular

selected_papers: true # includes a list of papers marked as "selected={true}"
social: true # includes social icons at the bottom of the page

announcements:
  enabled: true # includes a list of news items
  scrollable: false # adds a vertical scroll bar if there are more than 3 news items
  limit: 3 # leave blank to include all the news in the `_news` folder

latest_posts:
  enabled: true
  scrollable: false # adds a vertical scroll bar if there are more than 3 new posts items
  limit: 3 # leave blank to include all the blog posts
---

The **Calgary Machine Learning Lab**
is a research group led by [Yani Ioannou](https://yani.ai) within the [Schulich School of Engineering](https://schulich.ucalgary.ca) at the [University of Calgary](https://www.ucalgary.ca).
Our research is driven by the overarching goal of advancing efficient, trustworthy, and accessible Artificial Intelligence.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/labphotos/ictbuilding_may2026.jpg" loading="eager" fetchpriority="high" title="Schulich School of Engineering, University of Calgary" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Lab photo from May 2026 outside the ICT building at the University of Calgary.
</div>
Central to our work is the concept of sparse neural network training and inference, which we pursue with four key motivations:

- **Democratize AI**: removing redundant computation, making state-of-the-art models accessible to all
- **Sustainable and Trustworthy AI**: by fundamentally reducing the carbon footprint of these models while rigorously auditing how compression impacts algorithmic bias, we are working to ensure that real-world deployment of next-generation AI is both sustainable and safer
- **Learning Structure in Neural Networks**: automatically learn neural network topologies tailored for novel data domains
- **Understanding the Mechanics of Neural Network Training**: sparse neural network training provides a unique theoretical lens to **better understand** the underlying principles of neural network training, and its remarkable effectiveness.
