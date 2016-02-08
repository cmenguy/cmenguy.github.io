Title: Yet Another Data Blog in Pelican
Date: 2016-01-31 11:47
Tags: pelican, python, blog

Python and Data go very well together these days, so I've decided to move from Octopress to Pelican.
It was becoming a bit embarrassing not being able to hack the internals of my blog (not that I have anything against Ruby,
but it's just not one of my primary languages).
So after a gap of almost 2 years, I'm officially running on Pelican, and so far I'm thrilled about what it brings to the
table.

Why Pelican, you might ask? And what even is Pelican? Let me try to answer all of that in this inaugural post, which will I hope offer some insight to other folks looking at static websites.

# Why I said no to Octopress

So, like I said, Octopress being in Ruby has always held me down, but it wasn't the only issue. The whole transition to
Octopress 3.0 seemed a bit confusing to me, and I was reading many blog posts also complaining about it.

Feature\Engine | **Octopress** | **Pelican** | **Nikola**
--- | --------- | ------- | ------
**Community** | 9,300 stars, 3,100 forks | 5,400 stars, 1,200 forks | 920 stars, 250 forks
**Language** | Ruby | Python | Python
**Themes** | a lot | 100+ | 30+
**Deployment** | Mainly GH pages, more in 3.0 | GH pages, S3, Dropbox, FTP, CloudFiles, ... | Anything, but commands have to be specified manually
**IPython notebook support** | Plugin | Plugin | Native

I finally settled for Pelican, because it had a pretty good community behind it, a decent amount of themes available,
decent support for IPython notebook and, well, it's Python-based !

Nikola looked really good too, and some of their themes were outstanding, but I was a bit worried about the lack of
community behind it. Also the fact that I had trouble getting it to work on Python 2.7 made me a little bit worried.

# Crafting the right look & feel

One thing I've been struggling with Pelican, is finding the right theme for my blog.
The [pelican-bootstrap3](https://github.com/DandyDev/pelican-bootstrap3) blog looks pretty good, but I also really liked
the [BT3-flat](https://github.com/KenMercusLai/BT3-Flat), but I kinda wanted something in the middle, simple yet easy on
the eyes.

To that end I've extended the pelican-bootstrap3 theme to fit my needs. I'm really really not a front-end person so it
has been quite a bore, but I'm quite pleased with the final result (and hope I never have to touch this ever !)

First thing I wanted was the social buttons in the top bar instead of on the side - they look really out-of-place in 
bootstrap3, and to me felt much more natural at the top. For that, I modified **pelican-bootstrap3/templates/base.html**
like so:

    :::html
    {% if SOCIAL and INCLUDE_SOCIAL_IN_NAVBAR %}
        <!-- social media icons -->
        <ul class="nav navbar-nav navbar-right social">
            {% for s in SOCIAL %}
                {% if s[2] %}
                    {% set name_sanitized = s[2]|lower|replace('+','-plus')|replace(' ','-') %}
                {% else %}
                    {% set name_sanitized = s[0]|lower|replace('+','-plus')|replace(' ','-') %}
                {% endif %}
                {% set iconattributes = '"fa fa-lg fa-' ~ name_sanitized ~ '"' %}
                <li><a href="{{ s[1] }}"><i class={{ iconattributes }}></i></a></li>
            {% endfor %}
        </ul>
    {% endif %}

I also wanted to use the Pelican *about me* facilities, but found they rendered pretty poorly in bootstrap3.
For me a good *about me* element should contain the following:

* Personal information
* List of active tags
* A downloadable resume (we'll discuss that in more details in the next section)

I enclosed the *about me* section in a `well-sm` class in **pelican-bootstrap3/templates/includes/aboutme.html**:

    :::html
    <div id="aboutme" class="well well-sm">

Then inside that, I made it so every field can be filled from **pelicanconf.py** optionally:

    :::html
    {% if ABOUT_ME_TOPICS %}
        <div class="sub-name">{{ ABOUT_ME_TOPICS }}</div>
    {% endif %}
    {% if ABOUT_ME_EMAIL %}
        <div class="contact">{{ ABOUT_ME_EMAIL }}</div>
    {% endif %}
    {% if ABOUT_ME_LOCATION %}
        <div class="location">{{ ABOUT_ME_LOCATION }}</div>
    {% endif %}

At that point it's super easy to control your *about me* from the Pelican config. For example, here is what I have:

    :::python
    ABOUT_ME = True
    ABOUT_ME_TOPICS = "big data | data science | cloud computing"
    ABOUT_ME_EMAIL = "menguy.charles@gmail.com"
    ABOUT_ME_LOCATION = "New York City, United States"

Regarding the tags, I took them from the sidebar and added them in the *about me* section instead inline, which looked
more compact and aesthetically pleasing than in the sidebar to me:

    :::html
    {% if 'tag_cloud' in PLUGINS and DISPLAY_TAGS_ON_ABOUTME %}
        {% if DISPLAY_TAGS_INLINE %}
            {% set tags = tag_cloud | sort(attribute='0') %}
        {% else %}
            {% set tags = tag_cloud | sort(attribute='1') %}
        {% endif %}
        <br/>
        <ul class="list-group {% if DISPLAY_TAGS_INLINE %}list-inline tagcloud{% endif %}" id="tags">
            {% for tag in tag_cloud %}
                <li class="list-group-item tag-{{ tag.1 }}">
                    <a class="tag" href="{{ SITEURL }}/{{ tag.0.url }}">
                        #{{ tag.0 }}
                    </a>
                </li>
            {% endfor %}
        </ul>
    {% endif %}

And to top it off, I created a `CUSTOM_CSS` to adjust things a little bit and entered that in **pelicanconf.py**.

If you like the look & feel of this blog, feel free to reuse my code - I haven't made it a separate project because
it feels kind of an hybrid between two pre-existing themes, but if there is demand I can always create a separate
repo for it:

    git clone -b sources git@github.com:cmenguy/cmenguy.github.io.git

# Resume Automation

One thing that I wanted out of this blog is an easy, low-maintenance and automated way to update and share my resume.
An HTML is totally fine these days, but you also need a PDF resume in most cases, and that's where it gets tricky:
oftentimes they PDF and HTML get out-of-sync, creating the PDF goes through a different channel so you end up having
to replicate what you already wrote in HTML.

To solve that, I wrote a Pelican plugin called [pelican-resume](https://github.com/cmenguy/pelican-resume) to take care
of this automatically. With it, everytime you run `pelican content`, if you have a Markdown file under **pages/resume.md**
it will automatically create a PDF resume that can be embedded into your Pelican blog.

How to install it? Easy, with pip:

    pip install pelican-resume

How to use it? Simply include it in your list of plugins in **pelicanconf.py**:

    :::python
    PLUGINS = [
        # ...
        "pelican_resume",
        # ...
    ]

The default settings will take your **resume.md** and create a PDF resume using the [moderncv](https://www.ctan.org/pkg/moderncv)
style under **pdfs/resume.pdf** in your `OUTPUT_PATH`. All of this can be customized, and you are welcome to look at the
README in the repository for further information. Feel free to also contribute to it if you have custom CSS for different
resume styles, if you want to support something else than Markdown in input, ...

# Automation & Ease of use

One of the things I hated with Octopress, is the fact that I constantly ran into conflicts when deploying to master,
and it made me want to cry.

Here, I'm actually using [Travis](https://travis-ci.org/) to automatically and continuously deploy on every push without
having to deal with any of the manual work myself, which is a more than welcome change.

All of the code samples shown below are contained in the `.travis.yml` file which is placed at the root of the repository
in the branch containing your Pelican sources (I called mine **sources**).

The first thing we need to tell Travis is that it should only build in the **sources** branch, and not in master:

    :::yaml
    branches:
      only:
      - sources

Another thing I want is to be notified on every failure or success - I usually don't, but for a blog I like to be extra
sure that things are working ok when I push a new article.

    :::yaml
    notifications:
      email:
        on_success: always
        on_failure: always

Next the main issue I had with Travis was about installing everything required to build the Pelican blog. There's actually
a bunch of packages needed here:

* Python modules: pelican, markdown, beautifulsoup4, IPython, ghp-import and pelican-resume (see [requirements.txt](https://github.com/cmenguy/cmenguy.github.io/blob/sources/requirements.txt))
* Pandoc - somehow if we install it via Travis' `apt_packages` it uses an old version which produces poor formatting, so
we need to install it manually
* Wkhtmltopdf - this is not even available via `apt-get` for Travis, so it needs to be installed manually

The code snippet below contains the necessary commands to install all these dependencies on Travis:

    :::yaml
    install:
        # Install required Python modules
        - pip install -r requirements.txt
    
        # Install pandoc manually since it's disallowed with apt-get in Travis
        - mkdir $HOME/pandoc
        - curl -O https://s3.amazonaws.com/rstudio-buildtools/pandoc-1.12.3.zip
        - unzip -j pandoc-1.12.3.zip pandoc-1.12.3/linux/debian/x86_64/pandoc -d $HOME/pandoc
        - chmod +x $HOME/pandoc/pandoc
        - rm pandoc-1.12.3.zip
        - export PATH="$HOME/pandoc:$PATH"
        - pandoc --version
    
        # Install wkhtmltopdf manually since not available in apt-get
        - mkdir wkhtmltopdf
        - wget http://download.gna.org/wkhtmltopdf/0.12/0.12.3/wkhtmltox-0.12.3_linux-generic-amd64.tar.xz
        - tar --xz -xvf wkhtmltox-0.12.3_linux-generic-amd64.tar.xz -C wkhtmltopdf
        - export PATH=$PATH:$PWD/wkhtmltopdf/wkhtmltox/bin/

At that point Travis has everything it needs to build the blog and it just needs to run a `make publish`. I also save
the log and place it in the output so I can easily see what happened in case something goes wrong:

    :::yaml
    script:
    - pip freeze; make DEBUG=1 publish 2>&1 | tee -a build.log; cp build.log output/build.log

And finally it needs to publish to Github in case of success. This part is a little bit tricky, because we need to pass
an authentication token to Travis so it can push directly to the master branch. To do that I've modified the **Makefile**
to use the `GH_TOKEN` whose encrypted value is passed in **.travis.yml** via the `env.global.secure` field:

    :::yaml
    after_success: make github
    env:
      global:
        secure: <encrypted-gh-token>

To encrypt the token, you need to create one in the Github UI, and then encrypt it via the `travis` gem:

    gem install travis
    travis encrypt GH_TOKEN=<gh-token>

And voila ! With this simple configuration, you get a fully automated blog where you only need to fill your Markdown
pages, and Travis will take care of generating the relevant HTML and producing your PDF resume out of it.

----------------

With that system in place it should be extremely easy for me to keep updating my blog without wanting to kill myself
everytime I need to deploy something live.

Now that the boring UI thing is out of the way, I pledge to write at least 1 article per month on various data topics.
Some things I'm planning to focus on in this blog:

* Looking at Open Data, finding some insights and sharing them.
* Latest Big Data trends, and how they can be applied in a Data Science context.
* A new look at existing Big Data technologies.

Stay tuned for more !