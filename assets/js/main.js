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
});
