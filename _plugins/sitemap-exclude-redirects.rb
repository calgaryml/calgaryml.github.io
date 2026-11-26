# Exclude pages/documents that only redirect elsewhere (via `redirect:` front
# matter, rendered as a meta refresh in _layouts/default.liquid) from
# sitemap.xml. Google reports these as "Page with redirect" or "Discovered -
# currently not indexed" when they are submitted in the sitemap.
Jekyll::Hooks.register :site, :post_read do |site|
  items = site.pages + site.collections.values.flat_map(&:docs)
  items.each do |item|
    next unless item.data["redirect"]
    next if item.data.key?("sitemap")

    item.data["sitemap"] = false
  end
end
