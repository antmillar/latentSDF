$(document).ready(init);

function init(jQuery) {
  $("#btnGenerate").on('click', (e) => {
    document.querySelector("#slices").value = globals.slices;
});
}

window.globals = {
    generate: function() {},
    slices:  []
  }

