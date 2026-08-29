$(document).ready(function () {
  // Init Masonry for .projects and .places grid containers (excluding .people which uses CSS flex grid)
  $(".projects .grid, .places .grid").each(function () {
    var $grid = $(this).masonry({
      gutter: 10,
      horizontalOrder: true,
      itemSelector: ".grid-item",
    });
    // Layout Masonry after each image loads
    $grid.imagesLoaded().progress(function () {
      $grid.masonry("layout");
    });
  });
});
