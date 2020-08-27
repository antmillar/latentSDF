//populate values

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



//initialisation routine
$(document).ready(init);

function init(jQuery) {

  // generate model
  $("#gen").on('click', (e) => {

      if(window.globals.slices.length == 0)
        {
          alert("Cannot Build Model without a Path Defined");
        }
      else if( document.querySelector("#taper").value > 10.0)
      {
        alert("Cannot taper more than last 10% of slices");
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
        document.querySelector("#pathPoints").value = globals.points;
        document.querySelector("#modelHeight").value = globals.height;
        document.querySelector("#modelTaper").value = document.querySelector("#taper").value
        document.querySelector("#modelRotation").value = document.querySelector("#rotation").value
        document.querySelector("#show_context").value = $('#context').prop('checked')

        document.querySelector("#btnGenerate").submit();
      }
    
  });

  //update height
  $('#height').on('input', function() {
    window.globals.height = document.querySelector("#height").value;
    globals.getSlices(globals.height)
  });

  //update slices
  $('#slice_count').on('input', function() {
    window.globals.slice_count = parseInt(document.querySelector("#slice_count").value);
    console.log(window.globals.slice_count)
  });

  //update discrete on/off
  $('#discrete').on('change', function() {
    window.globals.discrete  = $('#discrete').prop('checked')
    console.log($('#discrete').prop('checked'))
  });

  //update constraint
  $("#constraint").on('click', (e) => {
      document.querySelector("#scoverage").value = window.globals.coverage;
      document.querySelector("#btnConstraint").submit();
  });

    //can only have one active constraint at time
  //update coverage value, untick site test
  $('#coverage').on('input', function() {

      window.globals.coverage = document.querySelector("#coverage").value;
      document.getElementById("constraint_site").checked = false
  });

  //if site test remove coverage value
  $("#constraint_site").on('change',  (e) => {

    document.querySelector("#ssite").value = $('#constraint_site').prop('checked')
    document.querySelector("#ssite_name").value = $('#site').prop('value')

    document.querySelector("#coverage").value = "";
    window.globals.coverage = document.querySelector("#coverage").value;
  }  )

  //load torch model 
  $("#file").on('change', (e) => {

    document.querySelector("#loadData").submit();
  });


  //on click for latent space update
  $("#latent").on('click', (e) => {
    document.querySelector("#latentForm").submit();
  });

  //on click to update design analysis
  $("#analysis").on('click', (e) => {
    
    document.querySelector("#launchAnalysis").submit();
  });
}