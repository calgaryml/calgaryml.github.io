---
# Template lab member profile: NOT published (Jekyll skips files starting with "_").
#
# To add yourself:
#   1. Copy this file to _people/<firstname>_<lastname>.md (lowercase, e.g. _people/jane_doe.md)
#   2. Add a square photo (at least 400x400, .jpg) to assets/img/people/ and set `img` below
#   3. Fill in the fields, delete the ones you don't use, and remove these instructions
#   4. Check it locally on the People page (/labmembers/) before opening a pull request
#
# Shared details about you go at the top; everything about a particular position in the lab
# (MSc, PhD, summer internship, visit, ...) goes in `roles`.
layout: page
firstname: Jane
lastname: Doe
pronouns: she/her # optional
img: assets/img/people/jane_doe.jpg
redirect: https://janedoe.github.io # optional: where clicking your photo/name goes (personal site, GitHub, LinkedIn)

# Optional links, each shown as an icon on your card; use just the username/ID, not the full URL
email: jane.doe@ucalgary.ca
github_username: janedoe
linkedin_username: jane-doe-123456
scholar_userid: AbCdEfGhIjK # from your Google Scholar profile URL (?user=...)
orcid_id: 0000-0000-0000-0000
twitter_username: janedoe
# Also supported: gitlab_username, medium_username, quora_username, keybase_username,
# stackoverflow_id, strava_userid, publons_id, research_gate_profile, wikidata_id,
# dblp_url, blogger_url, work_url

# After you leave the lab, where you are now (e.g. "Machine Learning Engineer, Example Corp").
# If left out, it's filled in automatically from your Google Scholar or GitHub profile.
# current_position: Machine Learning Engineer, Example Corp

show: true # set to false to hide yourself from the People page

# One entry per position in the lab, most recent first.
#   description: your position, as shown on the site (may contain HTML links, e.g. to a scholarship)
#   affiliation: visitors only, your home position and institution
#   category:    the People page section; one of: Lab Director, Postdoctoral Fellows, PhD Students,
#                MSc Students, Undergraduates, High School Students
#   start_term / start_year: when you started (terms: Winter, Spring, Summer, Fall)
#   expected_end_term / expected_end_year: optional, when you expect to finish
#   end_term / end_year: ONLY once the role has finished; this moves it to the Alumni list
roles:
  - description: MSc Student
    category: MSc Students
    start_term: Fall
    start_year: 2026
    expected_end_term: Summer
    expected_end_year: 2028
  # A finished role, e.g. a summer internship before starting your MSc:
  - description: <a href="https://www.nserc-crsng.gc.ca/students-etudiants/ug-pc/usra-brpc_eng.asp">NSERC USRA</a> Research Intern
    category: Undergraduates
    start_term: Summer
    start_year: 2025
    end_term: Summer
    end_year: 2025
  # A visit from another institution:
  # - description: Visiting Student Researcher
  #   affiliation: PhD Student, University of British Columbia
  #   category: PhD Students
  #   start_term: Winter
  #   start_year: 2026
---
