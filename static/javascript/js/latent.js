document.querySelector("#xMin").value = latent_bounds[0]
document.querySelector("#xMax").value = latent_bounds[1]
document.querySelector("#yMin").value = latent_bounds[2]
document.querySelector("#yMax").value = latent_bounds[3]
document.querySelector("#height").value = height

console.log(model_details)

$(document).ready(init);

function init(jQuery) {
  $("#gen").on('click', (e) => {

      if(window.globals.slices.length == 0)
        {
          alert("Cannot Build Model without a Path Defined");
        }
      else
      {
        document.querySelector("#slices").value = globals.slices;
        document.querySelector("#modelHeight").value = globals.height;
        document.querySelector("#modelTaper").value = document.querySelector("#taper").value
        document.querySelector("#btnGenerate").submit();
      }
    
});


  $('#height').on('input', function() {
    window.globals.height = document.querySelector("#height").value;
    globals.internalClicked(globals.height)
  });



    $("#constraint").on('click', (e) => {
  

      document.querySelector("#scoverage").value = globals.coverage;
      document.querySelector("#btnConstraint").submit();
      
  });

  $('#coverage').on('input', function() {
    window.globals.coverage = document.querySelector("#coverage").value;
  });
  }