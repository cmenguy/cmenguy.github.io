// Copy-to-clipboard for code blocks
document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll("div.highlight").forEach(function (block) {
    var wrapper = document.createElement("div");
    wrapper.className = "code-block-wrapper";
    block.parentNode.insertBefore(wrapper, block);
    wrapper.appendChild(block);

    var btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.textContent = "copy";
    btn.addEventListener("click", function () {
      var code = block.querySelector("pre").innerText;
      navigator.clipboard.writeText(code).then(function () {
        btn.textContent = "copied!";
        setTimeout(function () {
          btn.textContent = "copy";
        }, 2000);
      });
    });
    wrapper.appendChild(btn);
  });

  // Smooth anchor scrolling
  document.querySelectorAll('a[href^="#"]').forEach(function (anchor) {
    anchor.addEventListener("click", function (e) {
      var target = document.querySelector(this.getAttribute("href"));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: "smooth" });
      }
    });
  });

  // Table of Contents generation and scroll spy
  (function () {
    var tocNav = document.getElementById("toc");
    var tocList = tocNav ? tocNav.querySelector(".toc-list") : null;
    var postContent = document.querySelector(".post-content");
    if (!tocNav || !tocList || !postContent) return;

    var headings = postContent.querySelectorAll("h2, h3");
    if (headings.length < 2) {
      tocNav.style.display = "none";
      return;
    }

    // Build nested TOC structure with heading numbers
    var currentH2Item = null;
    var currentSubList = null;
    var h2Count = 0;
    var h3Count = 0;

    headings.forEach(function (heading) {
      if (!heading.id) return;

      var number;
      if (heading.tagName === "H2") {
        h2Count++;
        h3Count = 0;
        number = h2Count + ".";
      } else {
        h3Count++;
        number = h2Count + "." + h3Count;
      }

      var link = document.createElement("a");
      link.className = "toc-link";
      link.href = "#" + heading.id;

      var numSpan = document.createElement("span");
      numSpan.className = "toc-number";
      numSpan.textContent = number;
      link.appendChild(numSpan);
      link.appendChild(document.createTextNode(" " + heading.textContent));

      link.setAttribute("data-heading-id", heading.id);

      var li = document.createElement("li");
      li.appendChild(link);

      if (heading.tagName === "H2") {
        tocList.appendChild(li);
        currentH2Item = li;
        currentSubList = null;
      } else if (heading.tagName === "H3") {
        if (!currentH2Item) {
          tocList.appendChild(li);
        } else {
          if (!currentSubList) {
            currentSubList = document.createElement("ul");
            currentH2Item.appendChild(currentSubList);
          }
          currentSubList.appendChild(li);
        }
      }
    });

    // Scroll spy with IntersectionObserver
    var tocLinks = tocNav.querySelectorAll(".toc-link");
    var headingMap = {};
    tocLinks.forEach(function (link) {
      headingMap[link.getAttribute("data-heading-id")] = link;
    });

    var activeLink = null;

    function setActive(id) {
      if (activeLink) activeLink.classList.remove("active");
      if (headingMap[id]) {
        headingMap[id].classList.add("active");
        activeLink = headingMap[id];
        // Scroll TOC to keep active item visible
        activeLink.scrollIntoView({ block: "nearest", behavior: "smooth" });
      }
    }

    var observer = new IntersectionObserver(
      function (entries) {
        // Find the topmost visible heading
        var visibleEntries = entries.filter(function (e) {
          return e.isIntersecting;
        });
        if (visibleEntries.length > 0) {
          // Pick the one closest to the top of the viewport
          visibleEntries.sort(function (a, b) {
            return a.boundingClientRect.top - b.boundingClientRect.top;
          });
          setActive(visibleEntries[0].target.id);
        }
      },
      { rootMargin: "0px 0px -70% 0px", threshold: 0 }
    );

    headings.forEach(function (heading) {
      if (heading.id) observer.observe(heading);
    });

    // Set initial active heading on load
    var firstVisible = null;
    headings.forEach(function (heading) {
      if (!firstVisible && heading.getBoundingClientRect().top >= 0) {
        firstVisible = heading;
      }
    });
    if (firstVisible && firstVisible.id) {
      setActive(firstVisible.id);
    } else if (headings[0] && headings[0].id) {
      setActive(headings[0].id);
    }
  })();
});
