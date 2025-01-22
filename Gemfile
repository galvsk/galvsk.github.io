source "https://rubygems.org"

# This will include all the necessary plugins for GitHub Pages
gem "github-pages", group: :jekyll_plugins
gem "faraday-retry"

# Required plugins for minimal-mistakes theme
group :jekyll_plugins do
  gem "jekyll-feed", "~> 0.12"
  gem "jekyll-include-cache"
  gem "jekyll-paginate"
  gem "jekyll-sitemap"
  gem "jekyll-gist"
end

# Platform-specific gems
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
  gem "wdm", "~> 0.1"
  gem "http_parser.rb", "~> 0.6.0"
end

# Required for newer Ruby versions
gem "webrick"
