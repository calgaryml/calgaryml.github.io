#!/usr/bin/env ruby

require "optparse"
require "yaml"

ROOT = File.expand_path("..", __dir__)

options = { output: File.join(ROOT, "_data", "coauthors.yml") }
OptionParser.new do |parser|
  parser.banner = "Usage: generate_coauthors.rb [options]"
  parser.on("-o", "--output PATH", "Write the generated file to PATH") do |path|
    options[:output] = File.expand_path(path, Dir.pwd)
  end
end.parse!

def front_matter(path)
  content = File.read(path)
  match = content.match(/\A---\s*\n(.*?)\n---\s*\n/m)
  return unless match

  YAML.safe_load(match[1], permitted_classes: [], aliases: false)
end

def coauthor_key(lastname)
  lastname.split.last
          .unicode_normalize(:nfkd)
          .encode("ASCII", invalid: :replace, undef: :replace, replace: "")
          .downcase
end

def initials(firstname)
  parts = firstname.scan(/[[:alpha:]]+/)
  return "" if parts.empty?

  "#{parts.map { |part| part[0].upcase }.join(".")}."
end

data = YAML.safe_load_file(options[:output], permitted_classes: [], aliases: false) || {}

Dir[File.join(ROOT, "_people", "*.md")].sort.each do |path|
  person = front_matter(path)
  next unless person && person["show"] != false
  next if person["category"] == "Lab Director"
  next unless person["firstname"] && person["lastname"]

  key = coauthor_key(person["lastname"])
  firstname = person["firstname"]
  entries = data[key] ||= []
  entry = entries.find do |candidate|
    candidate["firstname"].any? do |name|
      name == firstname || name.start_with?("#{firstname} ")
    end
  end

  unless entry
    entries << {
      "firstname" => [firstname, initials(firstname)].uniq,
      "url" => "/labmembers/"
    }
  end
end

generated = YAML.dump(data)
File.write(options[:output], generated) unless File.exist?(options[:output]) && File.read(options[:output]) == generated