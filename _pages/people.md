---
layout: page
title: people
permalink: /labmembers/
# description: Lab Members
nav: true
nav_order: 3
# display_categories: [Lab Director, Postdoctoral Fellows, PhD Students, MSc Students, Research Associates, Undergraduates, Collaborators, Alumni]
# display_categories: [Lab Director, PhD Students, MSc Students, Research Assistants, Undergraduates, High School, Alumni]
display_categories: [Lab Director, PhD Students, MSc Students, Undergraduates]
# Each person in _people/ has a `roles` list; a role is shown under its `category` while current
# (no end_year; a planned end goes in expected_end_term/expected_end_year), and once it has an
# end_year it moves to the compact Alumni list at the bottom, grouped by end_year (most recent
# first) and, within a year, ordered by category in this order. A visitor's home position/institution
# goes in the role's `affiliation` (e.g. "PhD Student, University of British Columbia")
alumni_categories: [Postdoctoral Fellows, PhD Students, MSc Students, Undergraduates, High School Students]
horizontal: false
---

<div class="row align-items-center">
    <div class="col-sm">
      <div class="col-sm mt-3 mt-md-0">
          {% include figure.liquid path="assets/img/labphotos/schulich_may2026.jpg" loading="eager" fetchpriority="high" title="Schulich School of Engineering, University of Calgary" class="img-fluid rounded z-depth-1" %}
          <div class="caption">
              Schulich School of Engineering, University of Calgary (May. 2026)
          </div>
      </div>
      <div class="col-sm mt-3 mt-md-0">
          {% include figure.liquid path="assets/img/labphotos/lab_schulich_dec2025.jpg" title="Schulich School of Engineering, University of Calgary" class="img-fluid rounded z-depth-1" %}
          <div class="caption">
              Schulich School of Engineering, University of Calgary (Dec. 2025)
          </div>
      </div>
      <div class="col-sm mt-3 mt-md-0">
          {% include figure.liquid path="assets/img/labphotos/lab_schulich_april2024.jpg" title="Schulich School of Engineering, University of Calgary" class="img-fluid rounded z-depth-1" %}
          <div class="caption">
              Schulich School of Engineering, University of Calgary (Apr. 2024)
          </div>
      </div>
    </div>
    <div class="col-sm">

      <div class="col-sm mt-3 mt-md-0">
          {% include figure.liquid path="assets/img/labphotos/icml2026.jpg" title="ICML 2026" class="img-fluid rounded z-depth-1" %}
          <div class="caption">
              ICML 2026, Seoul, South Korea (July 2026)
          </div>
      </div>
      <div class="col-sm mt-3 mt-md-0">
          {% include figure.liquid path="assets/img/labphotos/iclr2026.jpg" title="ICLR 2026" class="img-fluid rounded z-depth-1" %}
          <div class="caption">
              ICLR 2026, Rio de Janeiro, Brazil (Apr. 2026)
          </div>
      </div>
      <div class="col-sm mt-3 mt-md-0">
          {% include figure.liquid path="assets/img/labphotos/icml2025.jpg" title="ICML 2025" class="img-fluid rounded z-depth-1" %}
          <div class="caption">
              ICML 2025, Vancouver, BC, Canada (July 2025)
          </div>
      </div>
      <div class="col-sm mt-3 mt-md-0">
          {% include figure.liquid path="assets/img/labphotos/neurips2024.jpg" title="NeurIPS 2024" class="img-fluid rounded z-depth-1" %}
          <div class="caption">
              NeurIPS 2024, Vancouver, BC, Canada (Dec. 2024)
          </div>
      </div>
      <div class="col-sm mt-3 mt-md-0">
          {% include figure.liquid path="assets/img/labphotos/peyto2023.jpg" title="Peyto Lake, Banff National Park" class="img-fluid rounded z-depth-1" %}
          <div class="caption">
              Peyto Lake, Banff National Park, AB, Canada (Aug. 2023)
          </div>
      </div>
    </div>

</div>

# Lab Members

<!-- pages/people.md -->
<div class="people">
  {%- assign people = site.people | where: "show", true | sort: "lastname" -%}
  <!-- Display categorized people -->
  {%- for category in page.display_categories %}
  <h2 class="category">{{ category }}</h2>
  <!-- Generate cards for each person's current role(s) in this category -->
  <div class="grid">
    {%- for person in people -%}
      {%- for role in person.roles -%}
        {%- if role.end_year == nil and role.category == category -%}
          {% include people.liquid %}
        {%- endif -%}
      {%- endfor -%}
    {%- endfor %}
  </div>
  {% endfor %}
</div>

# Alumni

<div class="people alumni">
  <p class="alumni-intro">Former lab members, by the year they finished.</p>
  {%- comment -%}
    One entry per finished role (one with an end_year), grouped by end_year, most recent first.
    Within a year, roles are ordered by page.alumni_categories (any other category last), then last name.
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
  {%- for year in end_years %}
  <div class="alumni-year-group">
    <h3 class="alumni-year">{{ year }}</h3>
    <ul class="alumni-list">
      {%- for category in category_order -%}
        {%- for person in people -%}
          {%- for role in person.roles -%}
            {%- comment -%} Current roles have no end_year, so they never match {%- endcomment -%}
            {%- assign role_year = role.end_year | append: "" -%}
            {%- if role_year != year -%}{%- continue -%}{%- endif -%}
            {%- if category == "*" -%}
              {%- if page.alumni_categories contains role.category -%}{%- continue -%}{%- endif -%}
            {%- elsif role.category != category -%}
              {%- continue -%}
            {%- endif -%}
            {% include alumni.liquid %}
          {%- endfor -%}
        {%- endfor -%}
      {%- endfor -%}
    </ul>
  </div>
  {%- endfor %}
</div>
