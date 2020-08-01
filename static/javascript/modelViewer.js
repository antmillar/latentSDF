import * as THREE from 'https://threejsfundamentals.org/threejs/resources/threejs/r113/build/three.module.js';
import View from './js/view.js';
import Model from './js/model.js';
import Controller from './js/gui.js';
import Scene from './js/scene.js';

//update navbar color on load
document.querySelector(".navbar").className = "navbar navbar-expand-lg navbar-light fixed-top"

//update the status
statusUpdate()

var mouse = new THREE.Vector2();
var view = new View(mouse);
var model = new Model();

var controller = new Controller(view, model);

//these arrays are loaded in the HTML template

model.loadScenes(inputFiles, outputFiles, meshFiles);

window.addEventListener( 'mousemove', onMouseMove, false );

//render loop
requestAnimationFrame(() => view.render());


//updates the status bar
function statusUpdate() {

    $.get('progress/' + 1, function(data) {
  
        document.querySelector("#status").textContent = "status : " + data;
        setTimeout(statusUpdate, 500);
  
    })
  }

//get latest mouse for raycasting
function onMouseMove( event ) {

    //normalized device coords (-1 to 1)
    mouse.x = ( (event.offsetX) / window.innerWidth ) * 2 - 1;
    mouse.y = -( (event.offsetY) / window.innerHeight ) * 2 + 1;

}


