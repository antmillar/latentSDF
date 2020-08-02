$(document).ready(init);


document.querySelector("#xMin").value = latentBounds[0]
document.querySelector("#xMax").value = latentBounds[1]
document.querySelector("#yMin").value = latentBounds[2]
document.querySelector("#yMax").value = latentBounds[3]


function init(jQuery) {
  $("#btnGenerate").on('click', (e) => {
    document.querySelector("#slices").value = globals.slices;
});
}

window.globals = {
    generate: function() {},
    slices:  [],
    latentBounds: latentBounds 
  }

