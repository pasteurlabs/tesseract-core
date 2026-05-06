// Open external links in a new tab
document.addEventListener("DOMContentLoaded", function () {
  var links = document.querySelectorAll("a[href]");
  var host = window.location.host;
  links.forEach(function (link) {
    if (link.hostname && link.hostname !== host) {
      link.setAttribute("target", "_blank");
      link.setAttribute("rel", "noopener noreferrer");
    }
  });
});
