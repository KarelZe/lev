// MathJax configuration for the formulas rendered by `pymdownx.arithmatex`
// in "generic" mode, which wraps them in elements of class `arithmatex`.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

// Re-typeset after instant navigation swaps the page content.  The bundle
// loads async, so on the first page it may not have replaced the config
// above yet; it typesets that page itself once it starts.
document$.subscribe(() => {
  if (!MathJax.startup?.output) return;
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
