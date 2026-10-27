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
# Alumni are listed compactly at the bottom, grouped by their alumni_category (role while in the lab)
# and sorted by end_year (year they finished), most recent first
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
  <!-- Display categorized people -->
  {%- for category in page.display_categories %}
  <h2 class="category">{{ category }}</h2>
  {%- assign categorized_people = site.people | where: "category", category -%}
  {%- assign sorted_people = categorized_people | sort: "lastname" %}
  <!-- Generate cards for each person -->
  <div class="grid">
    {%- for person in sorted_people -%}
      {%- if person.show -%}
        {% include people.liquid %}
      {%- endif -%}
    {%- endfor %}
  </div>
  {% endfor %}
</div>

# Alumni

<div class="people alumni">
  <p class="alumni-intro">Former lab members, grouped by their role in the lab and the year they finished.</p>
  {%- assign alumni = site.people | where: "category", "Alumni" | where: "show", true -%}
  {%- for category in page.alumni_categories %}
  {%- assign categorized_alumni = alumni | where: "alumni_category", category | sort: "lastname" -%}
  {%- if categorized_alumni.size > 0 %}
  <h2 class="category">{{ category }}</h2>
  {%- comment -%} Grouped by end_year, most recent first, then by last name; anyone without an end_year goes last {%- endcomment -%}
  {%- assign end_years = categorized_alumni | map: "end_year" | compact | uniq | sort | reverse -%}
  {%- for year in end_years %}
  <div class="alumni-year-group">
    <h3 class="alumni-year">{{ year }}</h3>
    <ul class="alumni-list">
      {%- assign year_alumni = categorized_alumni | where: "end_year", year -%}
      {%- for person in year_alumni -%}
        {% include alumni.liquid %}
      {%- endfor %}
    </ul>
  </div>
  {%- endfor %}
  {%- assign undated_alumni = categorized_alumni | where_exp: "person", "person.end_year == nil" -%}
  {%- if undated_alumni.size > 0 %}
  <div class="alumni-year-group">
    <h3 class="alumni-year">Other</h3>
    <ul class="alumni-list">
      {%- for person in undated_alumni -%}
        {% include alumni.liquid %}
      {%- endfor %}
    </ul>
  </div>
  {%- endif %}
  {%- endif %}
  {%- endfor %}
</div>
