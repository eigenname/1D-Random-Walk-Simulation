// Dropdown behavior for the top nav. Click-based (not hover-only) so it
// works on touch devices, and only one dropdown is designed to be open
// at a time in anticipation of more top-level tabs being added later.

document.addEventListener("DOMContentLoaded", () => {
  const toggles = document.querySelectorAll(".nav-dropdown-toggle");

  toggles.forEach((toggle) => {
    toggle.addEventListener("click", (event) => {
      event.stopPropagation();
      const isOpen = toggle.getAttribute("aria-expanded") === "true";
      toggles.forEach((t) => t.setAttribute("aria-expanded", "false"));
      toggle.setAttribute("aria-expanded", String(!isOpen));
    });
  });

  document.addEventListener("click", () => {
    toggles.forEach((t) => t.setAttribute("aria-expanded", "false"));
  });
});
