$(document).ready(init);


document.querySelector("#xMin").value = latentBounds[0]
document.querySelector("#xMax").value = latentBounds[1]
document.querySelector("#yMin").value = latentBounds[2]
document.querySelector("#yMax").value = latentBounds[3]


window.globals = {
    generate: function() {},
    slices:  [],
    latentBounds: latentBounds 
  }

  function init(jQuery) {
    $("#gen").on('click', (e) => {

        if(window.globals.slices.length == 0)
          {
            alert("Cannot Build Model without a Path Defined");
          }
        else
        {
          document.querySelector("#slices").value = globals.slices;
          document.querySelector("#btnGenerate").submit();
        }
      
  });
  }