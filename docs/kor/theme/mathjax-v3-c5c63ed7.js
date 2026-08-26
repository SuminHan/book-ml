// Replaces mdBook's built-in MathJax v2 (TeX-AMS-MML_HTMLorMML, HTML-CSS
// output) with MathJax v3 SVG output. v2's HTML-CSS output relies on
// separate webfont files (e.g. the calligraphic font used by \mathcal)
// fetched over the network -- if that font request is slow/blocked, the
// glyphs for that font silently fall back to tofu boxes while the rest of
// the page renders fine. SVG output draws every glyph as vector paths
// bundled directly in the JS, so there's no separate font file that can
// fail to load.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]]
  },
  svg: { fontCache: "global" }
};
(function () {
  var s = document.createElement("script");
  s.src = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg.js";
  s.async = true;
  document.head.appendChild(s);
})();
