#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = u'Charles Menguy'
SITENAME = u'Gradient Dissent'
SITEURL = 'http://localhost:8000'
SITELOGO = "images/logo2_black.jpg"
SITELOGO_SIZE = 40
HIDE_SITENAME = False
HIDE_SITELOGO = True

ABOUT_ME = True
ABOUT_ME_TOPICS = "big data | data science | cloud computing"
ABOUT_ME_EMAIL = "menguy.charles@gmail.com"
ABOUT_ME_LOCATION = "New York City, United States"

# AVATAR = "images/avatar.jpg" # a bit distracting, also doesn't resolve well with relative paths

PATH = 'content'

IGNORE_FILES = ['plugins/*', 'themes/*']

# plugins
PLUGIN_PATHS = ["plugins"]
PLUGINS = ['liquid_tags.img', 'liquid_tags.video',
           'liquid_tags.youtube', 'liquid_tags.vimeo',
           'liquid_tags.include_code', 'liquid_tags.notebook',
           'tipue_search', 'tag_cloud', 'pelican_resume', 'pelican-bootstrapify']

RESUME_TYPE = "moderncv"

DISPLAY_TAGS_ON_SIDEBAR = False
DISPLAY_TAGS_ON_ABOUTME = True
DISPLAY_TAGS_INLINE = True

# theme choice
THEME = "themes/pelican-bootstrap3"
#THEME = "themes/BT3-Flat"

### theme-specific parameters - see themes/*/README.md
BOOTSTRAP_THEME = "cosmo" # see http://bootswatch.com/
PYGMENTS_STYLE = "native" # see http://pygments.org/demo/218030

#BOOTSTRAP_NAVBAR_INVERSE = True

BOOTSTRAP_FLUID = True # more space on page but a bit too narrow

SHOW_ARTICLE_AUTHOR = False
SHOW_ARTICLE_CATEGORY = False
SHOW_DATE_MODIFIED = False

DISPLAY_PAGES_ON_MENU = False
DISPLAY_CATEGORIES_ON_MENU = False
DISPLAY_RECENT_POSTS_ON_SIDEBAR = True

MENUITEMS = (
    ('About','/pages/about.html'),
    ("Resume", "/pages/resume.html"),
    ("Blog", "/index.html"),
    ("Feed", "feeds/all.atom.xml")
)

DIRECT_TEMPLATES = ('index', 'categories', 'authors', 'archives', 'search')

CUSTOM_CSS = 'static/custom.css'

STATIC_PATHS = [
    'images',
    'files',
    'extra/robots.txt',
    'extra/favicon.ico',
    'extra/custom.css'
]

EXTRA_PATH_METADATA = {
    'extra/robots.txt': {'path': 'robots.txt'},
    'extra/favicon.ico': {'path': 'favicon.ico'},
    'extra/custom.css': {'path': 'static/custom.css'}
}

BANNER = 'images/banner.jpg'
BANNER_SUBTITLE = "A different perspective on all things data"
###

# GITHUB_USER = 'cmenguy'
# GITHUB_SKIP_FORK = True

TWITTER_USERNAME = "cmenguy" # used to enable twitter in sidebar, not really useful
#TWITTER_WIDGET_ID = "693710186968715264"
TWITTER_CARDS = True

TIMEZONE = 'America/New_York'

DEFAULT_LANG = u'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
LINKS = (('Hacker News', 'https://news.ycombinator.com/'),
         ("Databricks Blog", "https://databricks.com/blog"),
         ('PyEvolve', 'http://blog.christianperone.com/'),
         ("Pythonic Perambulations", "https://jakevdp.github.io/"),
         ('UCI Machine Learning Repository', 'http://archive.ics.uci.edu/ml/'),
         ('My school', 'http://www.epita.fr/'),)

# Social widget
SOCIAL = (("linkedin", 'https://www.linkedin.com/in/charlesmenguy'),
          ("github", "https://github.com/cmenguy"),
          ("twitter", "https://twitter.com/cmenguy"),
          ("stackoverflow", "http://stackoverflow.com/users/1332690/charles-menguy", "stack-overflow"),
          ("infoq", "http://www.infoq.com/author/Charles-Menguy", "newspaper-o"),
          ("coursera", "https://www.coursera.org/user/i/e6c7809cb65307e8057090514dd4367e", "university"),
          ("kaggle", "https://www.kaggle.com/cmenguy", "bar-chart")
          )
INCLUDE_SOCIAL_IN_SIDEBAR = False
INCLUDE_SOCIAL_IN_NAVBAR = True

ADDTHIS_PROFILE = "ra-56ada96e76fa8aac"
DISQUS_SITENAME = "gradientdissent"
GOOGLE_ANALYTICS = "UA-45200068-2"

CC_LICENSE = "CC-BY-NC-SA"

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True

ARTICLE_URL = 'blog/{slug}.html'
ARTICLE_SAVE_AS = 'blog/{slug}.html'
PAGE_URL = 'pages/{slug}.html'
PAGE_SAVE_AS = 'pages/{slug}.html'
TAG_URL = 'tags/{slug}.html'
TAG_SAVE_AS = 'tags/{slug}.html'