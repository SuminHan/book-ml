(function () {
  var a = document.createElement("a");
  // Swap the /kor/ path segment for /eng/ so the badge lands on the same
  // chapter in the other language, not just the other book's front page.
  // Both books mirror the same file layout (ml1/chapterNN.md,
  // ml2/chapterNN.md, README.md), so a straight segment swap always maps
  // to the corresponding page.
  var target = window.location.pathname.replace(/\/kor(\/|$)/, "/eng$1") + window.location.hash;
  a.href = target;
  a.textContent = "🌐 English";
  a.className = "lang-switch-badge";
  document.body.appendChild(a);
})();
