require 'json'
require 'nokogiri'
require 'open-uri'

# Populates `current_position` for lab alumni (people whose roles all have an `end_year`)
# at build time, so the alumni list can show where they are now.
#
# Precedence, first non-empty wins:
#   1. `current_position` set manually in the person's front matter
#      (use this for LinkedIn info: LinkedIn blocks automated access)
#   2. Google Scholar profile affiliation (`scholar_userid`)
#   3. GitHub profile "company" field (`github_username`)
#
# Fetched values matching `stale_patterns` (e.g. the lab's own university) are
# ignored, since they usually mean the profile hasn't been updated since the
# person left. Set `current_position_auto: false` on a person to opt out.
# Results are cached in .jekyll-cache so local `jekyll serve` stays fast.
module Jekyll
  class AlumniPositionsGenerator < Generator
    safe true
    priority :high

    DEFAULTS = {
      'enabled' => true,
      'cache_days' => 7,
      'timeout' => 10,
      'stale_patterns' => [],
    }.freeze

    def generate(site)
      @config = DEFAULTS.merge(site.config['alumni_positions'] || {})
      return unless @config['enabled']

      people = site.collections['people']
      return unless people

      @cache_file = File.join(site.source, '.jekyll-cache', 'alumni-positions.json')
      @cache = load_cache

      people.docs.each do |person|
        data = person.data
        # Only people who have left: every role finished (has an end_year)
        roles = Array(data['roles'])
        next if roles.empty? || !roles.all? { |role| role['end_year'] }
        next if data['current_position'].to_s.strip != ''
        next if data['current_position_auto'] == false

        position = lookup("scholar:#{data['scholar_userid']}") { scholar_affiliation(data['scholar_userid']) } if data['scholar_userid']
        position = nil if stale?(position)
        if position.nil? && data['github_username']
          position = lookup("github:#{data['github_username']}") { github_company(data['github_username']) }
          position = nil if stale?(position)
        end
        data['current_position'] = position if position
      end

      save_cache
    end

    private

    def stale?(position)
      return false if position.nil?
      @config['stale_patterns'].any? { |pattern| position.match?(Regexp.new(pattern, Regexp::IGNORECASE)) }
    end

    # Returns a cached value if fresh, otherwise fetches (failures are cached
    # too, as nil, so an offline build doesn't retry every regeneration).
    def lookup(key)
      entry = @cache[key]
      if entry && Time.now.to_i - entry['fetched_at'] < @config['cache_days'] * 86_400
        return entry['value']
      end

      value =
        begin
          yield
        rescue StandardError => e
          Jekyll.logger.warn 'Alumni positions:', "could not fetch #{key}: #{e.class} - #{e.message}"
          # Keep a previously fetched value rather than dropping it on a transient error
          entry ? entry['value'] : nil
        end
      @cache[key] = { 'value' => value, 'fetched_at' => Time.now.to_i }
      value
    end

    def scholar_affiliation(scholar_id)
      # Sleep for a random amount of time to avoid being blocked
      sleep(rand(1.0..2.5))
      url = "https://scholar.google.com/citations?user=#{scholar_id}&hl=en"
      doc = Nokogiri::HTML(URI.open(url, 'User-Agent' => 'Mozilla/5.0', read_timeout: @config['timeout'], open_timeout: @config['timeout']))
      affiliation = doc.at_css('#gsc_prf_i .gsc_prf_il')&.text.to_s.squeeze(' ').strip
      affiliation.empty? ? nil : affiliation
    end

    def github_company(username)
      headers = { 'Accept' => 'application/vnd.github+json', 'User-Agent' => 'calgaryml-site' }
      headers['Authorization'] = "Bearer #{ENV['GITHUB_TOKEN']}" if ENV['GITHUB_TOKEN']
      body = URI.open("https://api.github.com/users/#{username}", headers.merge(read_timeout: @config['timeout'], open_timeout: @config['timeout'])).read
      company = JSON.parse(body)['company'].to_s.strip.delete_prefix('@')
      company.empty? ? nil : company
    end

    def load_cache
      File.exist?(@cache_file) ? JSON.parse(File.read(@cache_file)) : {}
    rescue StandardError
      {}
    end

    def save_cache
      FileUtils.mkdir_p(File.dirname(@cache_file))
      File.write(@cache_file, JSON.pretty_generate(@cache))
    rescue StandardError => e
      Jekyll.logger.warn 'Alumni positions:', "could not write cache: #{e.message}"
    end
  end
end
