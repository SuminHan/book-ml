(function () {
  // mdBook wraps every ![alt](src) image in a click-to-zoom
  // <label class="checkbox-label"><img alt="..."> ...</label>, but never
  // displays that alt text anywhere on the page. Since every reference
  // figure in this book carries its real caption as the alt text (see
  // tools/ref_library.json's figure_caption field), promote it to a
  // visible caption line under the image.
  document.querySelectorAll(".checkbox-label > img").forEach(function (img) {
    if (!img.alt) return;
    var host = img.closest("p") || img.closest(".checkbox-label");
    if (!host) return;
    if (host.nextElementSibling && host.nextElementSibling.classList.contains("ref-figure-caption")) return;
    var caption = document.createElement("p");
    caption.className = "ref-figure-caption";
    caption.textContent = img.alt;
    host.insertAdjacentElement("afterend", caption);
  });
})();
