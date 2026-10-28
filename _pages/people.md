---
layout: page
title: People
permalink: /labmembers/
description: Current members and alumni of the Calgary Machine Learning Lab
nav: true
nav_order: 3
# display_categories: [Lab Director, Postdoctoral Fellows, PhD Students, MSc Students, Research Associates, Undergraduates, Collaborators, Alumni]
# display_categories: [Lab Director, PhD Students, MSc Students, Research Assistants, Undergraduates, High School, Alumni]
display_categories: [Lab Director, PhD Students, MSc Students, Undergraduates]
# Each person in _people/ has a `roles` list; a role is shown under its `category` while current
# (no end_year; a planned end goes in expected_end_term/expected_end_year), and once it has an
# end_year it moves to the compact Alumni list at the bottom, grouped by end_year (most recent
# first) and, within a year, by end_term (most recent first) then by category in this order.
# A visitor's home position/institution goes in the role's `affiliation`
# (e.g. "PhD Student, University of British Columbia")
alumni_categories: [Postdoctoral Fellows, PhD Students, MSc Students, Undergraduates, High School Students]
# Lab photos, most recent first. lab_photos_layout is either
#   slideshow: all photos in a slideshow at the top of the page
#   banner:    the first photo at the top, the rest in a Lab Photos gallery at the bottom
# (images.slider loads the slideshow library; it's only needed for the slideshow layout)
lab_photos_layout: slideshow
images:
  slider: true
lab_photos:
  - path: assets/img/labphotos/icml2026.jpg
    caption: ICML 2026, Seoul, South Korea (July 2026)
  - path: assets/img/labphotos/schulich_may2026.jpg
    caption: Schulich School of Engineering, University of Calgary (May 2026)
  - path: assets/img/labphotos/iclr2026.jpg
    caption: ICLR 2026, Rio de Janeiro, Brazil (Apr. 2026)
  - path: assets/img/labphotos/lab_schulich_dec2025.jpg
    caption: Schulich School of Engineering, University of Calgary (Dec. 2025)
  - path: assets/img/labphotos/icml2025.jpg
    caption: ICML 2025, Vancouver, BC, Canada (July 2025)
  - path: assets/img/labphotos/neurips2024.jpg
    caption: NeurIPS 2024, Vancouver, BC, Canada (Dec. 2024)
  - path: assets/img/labphotos/lab_schulich_april2024.jpg
    caption: Schulich School of Engineering, University of Calgary (Apr. 2024)
  - path: assets/img/labphotos/peyto2023.jpg
    caption: Peyto Lake, Banff National Park, AB, Canada (Aug. 2023)
horizontal: false
---

{%- if page.lab_photos_layout == 'slideshow' %}

{% include lab_photos_slideshow.liquid %}

{%- else %}
{%- assign group_photo = page.lab_photos | first -%}

<div class="lab-group-photo">
  {%-
    include figure.liquid
    path=group_photo.path
    loading="eager"
    fetchpriority="high"
    alt=group_photo.caption
    caption=group_photo.caption
    class="img-fluid rounded z-depth-1"
    sizes="(min-width: 930px) 900px, 95vw"
  -%}
</div>
{%- endif %}

# Lab Members

<!-- pages/people.md -->
<div class="people">
  {%- assign people = site.people | where: "show", true | sort: "lastname" -%}
  <!-- One block per category, sized to its people, so small categories sit side by side -->
  <div class="category-groups">
    {%- for category in page.display_categories %}
      {%- capture cards -%}
        {%- for person in people -%}
          {%- for role in person.roles -%}
            {%- if role.end_year == nil and role.category == category -%}
              {% include people.liquid %}
            {%- endif -%}
          {%- endfor -%}
        {%- endfor -%}
      {%- endcapture -%}
      {%- if cards != "" %}
        <section class="category-group">
          <h2 class="category">{{ category }}</h2>
          <div class="grid">{{ cards }}</div>
        </section>
      {%- endif -%}
    {%- endfor %}
  </div>
</div>

# Alumni

<div class="people alumni">
  <p class="alumni-intro">Former lab members, ordered by term and year they finished.</p>
  {%- comment -%}
    One entry per finished role (one with an end_year), grouped by end_year, most recent first.
    Within a year, roles are ordered by end_term, most recent first (any without one last), then by
    page.alumni_categories (any other category last), then last name.
  {%- endcomment -%}
  {%- assign end_years = "" -%}
  {%- for person in people -%}
    {%- for role in person.roles -%}
      {%- if role.end_year -%}
        {%- assign end_years = end_years | append: role.end_year | append: "," -%}
      {%- endif -%}
    {%- endfor -%}
  {%- endfor -%}
  {%- assign end_years = end_years | split: "," | uniq | sort | reverse -%}
  {%- assign category_order = page.alumni_categories | push: "*" -%}
  {%- assign term_order = "Fall,Summer,Spring,Winter" | split: "," -%}
  {%- assign terms = term_order | push: "*" -%}
  {%- for year in end_years %}
  <div class="alumni-year-group">
    <h3 class="alumni-year">{{ year }}</h3>
    <ul class="alumni-list">
      {%- assign shown_term = nil -%}
      {%- for term in terms -%}
        {%- for category in category_order -%}
          {%- for person in people -%}
            {%- for role in person.roles -%}
              {%- comment -%} Current roles have no end_year, so they never match {%- endcomment -%}
              {%- assign role_year = role.end_year | append: "" -%}
              {%- if role_year != year -%}{%- continue -%}{%- endif -%}
              {%- if term == "*" -%}
                {%- if term_order contains role.end_term -%}{%- continue -%}{%- endif -%}
              {%- elsif role.end_term != term -%}
                {%- continue -%}
              {%- endif -%}
              {%- if category == "*" -%}
                {%- if page.alumni_categories contains role.category -%}{%- continue -%}{%- endif -%}
              {%- elsif role.category != category -%}
                {%- continue -%}
              {%- endif -%}
              {%- comment -%} The term is labelled once, on its first entry {%- endcomment -%}
              {%- if term != shown_term -%}
                {%- assign show_term = true -%}
                {%- assign shown_term = term -%}
              {%- else -%}
                {%- assign show_term = false -%}
              {%- endif -%}
              {% include alumni.liquid %}
            {%- endfor -%}
          {%- endfor -%}
        {%- endfor -%}
      {%- endfor -%}
    </ul>
  </div>
  {%- endfor %}
</div>

{% if page.lab_photos_layout != 'slideshow' %}

# Lab Photos

{% include lab_photos_gallery.liquid %}
{% endif %}
