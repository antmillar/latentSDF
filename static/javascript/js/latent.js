document.querySelector("#xMin").value = latent_bounds[0];
document.querySelector("#xMax").value = latent_bounds[1];
document.querySelector("#yMin").value = latent_bounds[2];
document.querySelector("#yMax").value = latent_bounds[3];
document.querySelector("#height").value = height;

document.querySelector("#c").height = canvas_height;
document.querySelector("#c").width = canvas_width;
// document.querySelector("#paperCanvas").height = canvas_height
// document.querySelector("#paperCanvas").height = canvas_width

document.querySelector("#coverage").value = coverage;
document.querySelector("#context").checked = show_context == "true";

console.log(model_details)

$(document).ready(init);

document.querySelector("#file").onchange = function() {

  document.querySelector("#loadData").submit();
};

function init(jQuery) {
  $("#gen").on('click', (e) => {

      if(window.globals.slices.length == 0)
        {
          alert("Cannot Build Model without a Path Defined");
        }
      else if( document.querySelector("#taper").value > globals.height / 10)
      {
        alert("Cannot taper more than last 10% of layers");
      }
      else if( document.querySelector("#taper").value < 0)
      {
        alert("Cannot have negative taper");
      }
      else if( document.querySelector("#slice_count").value < 1)
      {
        alert("Cannot have less than 1 slice per unit height");
      }
      else if(!Number.isInteger(window.globals.slice_count)) //COULD BE BETTER
      {
        alert("Slice per floor must be integer");

      }
      else
      {
        document.querySelector("#slices").value = globals.slices;
        document.querySelector("#modelHeight").value = globals.height;
        document.querySelector("#modelTaper").value = document.querySelector("#taper").value
        document.querySelector("#modelRotation").value = document.querySelector("#rotation").value
        document.querySelector("#show_context").value = $('#context').prop('checked')

        document.querySelector("#btnGenerate").submit();
      }
    
  });


  $('#height').on('input', function() {
    window.globals.height = document.querySelector("#height").value;
    globals.internalClicked(globals.height)
  });

  $('#slice_count').on('input', function() {
    window.globals.slice_count = parseInt(document.querySelector("#slice_count").value);
    console.log(window.globals.slice_count)
  });

  $('#discrete').on('change', function() {
    window.globals.discrete  = $('#discrete').prop('checked')
    console.log($('#discrete').prop('checked'))
  });




  $("#constraint").on('click', (e) => {
  
      document.querySelector("#scoverage").value = globals.coverage;
      document.querySelector("#btnConstraint").submit();
      
  });

  $('#coverage').on('input', function() {
    window.globals.coverage = document.querySelector("#coverage").value;
  });

//   $("#download").on('click', (e) => {

  
//     document.querySelector("#downloadModel").value = 1;
//     document.querySelector("#btnDownload").submit();
    
// });


$("#latent").on('click', (e) => {
  

  document.querySelector("#latentForm").submit();
  
});

$("#analysis").on('click', (e) => {
  

  document.querySelector("#launchAnalysis").submit();
  
});
}