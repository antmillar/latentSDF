import * as THREE from 'https://threejsfundamentals.org/threejs/resources/threejs/r113/build/three.module.js';
import {OrbitControls} from 'https://threejsfundamentals.org/threejs/resources/threejs/r113/examples/jsm/controls/OrbitControls.js';
import {OBJLoader2} from 'https://threejsfundamentals.org/threejs/resources/threejs/r113/examples/jsm/loaders/OBJLoader2.js';
import {STLExporter} from 'https://threejsfundamentals.org/threejs/resources/threejs/r113/examples/jsm/exporters/STLExporter.js';

var scene, camera, renderer, canvas;
var dloadInterior
var dloadExterior

canvas = document.querySelector('#c');
var fn_site = "/static/models/inputs/context_sm.obj"

  //scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xBBBBBB);

  //camera
  const fov = 45;
  const aspect = 1;  // the canvas default
  const clipNear = 0.01;
  const clipFar = 5000;
  const radius = 400
  camera = new THREE.PerspectiveCamera(fov, aspect, clipNear, clipFar);

  var angle = 1.25 * Math.PI / 4.0 ;
  // var angle = -1.25 * Math.PI / 4.0 ;

  camera.position.set(radius * Math.cos(angle) , radius * 0.6, radius * Math.sin(angle) );

  //mouse controls
  const controls = new OrbitControls(camera, canvas);
  controls.target.set(0, 50, 0);
  controls.update();

  var ambient = new THREE.AmbientLight( 0xbbbbbb, 0.5 );
  scene.add( ambient );

  scene.fog = new THREE.Fog( 0xBBBBBB, 500, 2000  );

  var dirLight = new THREE.DirectionalLight( 0xffffff, 0.5);
  dirLight.position.set( -100, 200, 100 );
  dirLight.castShadow = true;
  // dirLight.shadow.mapSize = new THREE.Vector2(512, 512);
  scene.add(dirLight);

  var dirLight2 = new THREE.DirectionalLight( 0xffffff, 0.5);
  dirLight2.position.set( 100, 200, -100 );
  dirLight2.castShadow = true;
  // dirLight.shadow.mapSize = new THREE.Vector2(512, 512);
  scene.add(dirLight2);

  renderer = new THREE.WebGLRenderer({preserveDrawingBuffer: true , antialias:true, canvas : canvas});
  let width = canvas_width;
  let height = canvas_height;
  // renderer.setPixelRatio( window.devicePixelRatio );
  renderer.setSize(width, height);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  if(active_model == "") 
  {
    
  }
  else
  {

  loadCore();
  
  // LOAD THE BUILDING, FLOORS and CONTOURS
  let loader = new OBJLoader2();
  loader.load(active_model, function(obj){

    let matGlass2 = new THREE.MeshPhysicalMaterial( {
      color: 0x555555,
      opacity: 0.5,
      side: THREE.FrontSide,
      transparent: true,

    } );

    //geometry
    let model_facade = obj.children[0];
    model_facade.name = "facade";
    // console.log("bldg model : " , model_facade)

    model_facade.geometry.computeVertexNormals();
    model_facade.material = matGlass2
    model_facade.castShadow = true;

    let ground =  new THREE.Mesh(
      new THREE.PlaneBufferGeometry( 5000, 5000 ),
      new THREE.MeshPhongMaterial( { color: 0xBBBBBB, side: THREE.DoubleSide } )
    );
    ground.receiveShadow = true;

    //scaling/rotating
    // model_facade.geometry.scale(0.1,0.1,0.1);
    model_facade.geometry.rotateZ( Math.PI / 2); 

    ground.rotation.x = -Math.PI / 2; 

    generateFloors();

    generateContours();

    generateInteriorModel(scene);

    scene.add(model_facade, ground);

    generateExteriorModel(model_facade);


  });

 
}

animate();





function animate() {

  renderer.render(scene,camera);
  requestAnimationFrame(animate);
}

function toggleContext(toggle)
{
  if(toggle)
  {

    
    console.log("adding context...")

    // LOAD THE CONTEXT OBJ FILE
    let contextLoader = new OBJLoader2();

    contextLoader.load(fn_site, function(obj){

      let matContext = new THREE.MeshPhysicalMaterial( {
        color: 0xAAAAAA,
      } );

      let model_context = obj.children[0];
      // console.log("context model : " , model_context)
      model_context.material = matContext;
      model_context.receiveShadow = true;
      let box = new THREE.Box3().setFromObject( model_context );
      let center = new THREE.Vector3();
      box.getCenter( center );

      // model_context.position.sub( center ); // center the model
      // model_context.geometry.scale(0.8,0.8,0.8);
      model_context.rotation.x = Math.PI/2;   // rotate the model
      model_context.scale

      //hard coded offsets for the specific context used
      // model_context.translateY(100)
      // model_context.translateX(-16)
      // model_context.translateZ(-50)

      model_context.name = "context"

      scene.add(model_context);

    })

    let objFacade = scene.getObjectByName("facade");

    console.log(objFacade)
    let objToRemove = scene.getObjectByName("axes");
    scene.remove(objToRemove)

  }

  else
  {
    console.log("removing context...")
    let objToRemove = scene.getObjectByName("context");
    scene.remove(objToRemove)

    let objFacade = scene.getObjectByName("facade");
    generateAxes(objFacade);

  }

  animate()

}


function loadCore(){

  // LOAD THE CORE OBJ FILE
  let coreLoader = new OBJLoader2();
  let fn_core = active_model.slice(0, -23) + "core_" + active_model.slice(-23)
  
  coreLoader.load(fn_core, function(obj){

    let matCore = new THREE.MeshPhysicalMaterial( {
      color: 0x111111,
      opacity:0.5,
      // side: THREE.DoubleSide,
      transparent: true,
    } );

    let model = obj.children[0];
    // console.log("core model : " , model)
     
    model.material = matCore;

    // model.geometry.scale(0.1,0.1,0.1);
    
    model.geometry.rotateZ( Math.PI / 2); 

    scene.add(model);

  })


}

function generateFloors(){


    let matGlass = new THREE.MeshPhysicalMaterial( {
      color: 0xDDDDDD,
      opacity: 0.75,
    } );

    
    //create floors in building
    for(let i = 0; i < floors.length; i++)
    {
    //skip first floor as double height
      if(i == 1)
      {
        continue;
      }


      //check if the floor contains more than one contour
      let floorContourCount = floors[i].length
      for(let j = 0; j < floorContourCount; j++)
      {      
        let points = [];
        for(let k = 0; k < floors[i][j].length; k++)
       {
         //get coordinates
          points.push(new THREE.Vector3(floors[i][j][k][0],floors[i][j][k][1],floors[i][j][k][2]))
        }

        let floorOutline = new THREE.Shape(points);
        let extrudeSettings = { depth:0.1, bevelEnabled: false};
  
        let geometryFloor = new THREE.ExtrudeBufferGeometry( floorOutline, extrudeSettings );
        let matEdge = new THREE.LineBasicMaterial( { color: 0x111111, linewidth:0.75} )
        let edges = new THREE.EdgesGeometry( geometryFloor);
        let edge = new THREE.LineSegments( edges, matEdge );

        // geometryFloor.scale(0.1,0.1,0.1);
        geometryFloor.rotateZ( Math.PI ); 
        geometryFloor.rotateX( -Math.PI / 2); 
        geometryFloor.translate(0,   points[0].z, 0);


        let floor = new THREE.Mesh( geometryFloor, matGlass );
        floor.castShadow = true;

        // edge.scale.set(0.1,0.1,0.1);
        edge.rotation.x = Math.PI / 2; 
        edge.rotation.y = Math.PI; 
        edge.translateZ(points[0].z);
  
        scene.add( floor, edge );
      }
    }

}


function generateContours()
{
    let matContours = new THREE.LineBasicMaterial( { color: 0x444444, linewidth: 0.5,   } );


    //create vertical contours in building
    for(let j = 0; j < contours.length; j++)
    {
      var points = [];
      
      // populate contour points
      for(let i = 0; i < contours[j][0].length; i++)
      {
        points.push(new THREE.Vector3(contours[j][0][i][0],contours[j][0][i][1],contours[j][0][i][2]))
      }

      points.push(points[0]) //close the loop
      let curve = new THREE.CatmullRomCurve3(points)
      let curvePts = curve.getPoints( 50 );
      let geometryContour = new THREE.BufferGeometry().setFromPoints( curvePts );
      
      let contourLine = new THREE.Line( geometryContour, matContours );
      // contourLine.geometry.scale(0.1,0.1,0.1);
      contourLine.geometry.rotateX( -Math.PI / 2); 
      contourLine.geometry.rotateY(Math.PI); 

      scene.add(contourLine)


    }

}

function generateAxes(model_facade){

  let bbox = new THREE.Box3().setFromObject( model_facade );
  let size = bbox.getSize();
  
  var xSize = (size.x ).toFixed(2);
  var zSize = (size.z ).toFixed(2);

  var bldg_height = window.globals.height;
  var matAxis = new THREE.LineBasicMaterial({
    color: 0x888888,
    linewidth: 5.0
  });
  // add dimension lines

  var buffer = 5
  var lenTip = 2.5

  var xPts = [];
  xPts.push( new THREE.Vector3( -xSize/2 + buffer, 10 , -50 - buffer + lenTip) );
  xPts.push( new THREE.Vector3( -xSize/2 + buffer, 10, -50 - buffer - lenTip) );
  xPts.push( new THREE.Vector3( -xSize/2 + buffer, 10, -50 - buffer) );
  xPts.push( new THREE.Vector3( xSize/2 - buffer, 10 , -50 - buffer) );
  xPts.push( new THREE.Vector3( xSize/2 - buffer, 10 , -50 - buffer + lenTip) );
  xPts.push( new THREE.Vector3( xSize/2 - buffer, 10 , -50 - buffer - lenTip) );

  var geoX = new THREE.BufferGeometry().setFromPoints( xPts );
  var xAxis = new THREE.Line( geoX, matAxis );


  var zPts = [];

  zPts.push( new THREE.Vector3( -50 + lenTip, 10, zSize/2  - buffer) );
  zPts.push( new THREE.Vector3( -50 - lenTip, 10, zSize/2  - buffer) );
  zPts.push( new THREE.Vector3( -50 , 10, zSize/2  - buffer) );
  zPts.push( new THREE.Vector3( -50 , 10 , -zSize/2 + buffer) )
  zPts.push( new THREE.Vector3( -50  + lenTip, 10 , -zSize/2 + buffer) )
  zPts.push( new THREE.Vector3( -50  - lenTip, 10 , -zSize/2 + buffer) )
  

  var geoZ = new THREE.BufferGeometry().setFromPoints( zPts );
  var zAxis = new THREE.Line( geoZ, matAxis );


  var yPts = [];
  yPts.push( new THREE.Vector3( -50 - lenTip, bldg_height  + buffer  ,-50 - lenTip) )
  yPts.push( new THREE.Vector3( -50 + 2*lenTip, bldg_height  + buffer ,-50 + 2*lenTip) )

  
  var geoY = new THREE.BufferGeometry().setFromPoints( yPts );
  var yAxis = new THREE.Line( geoY, matAxis );



  // adding the axis labels

  var fontLoader = new THREE.FontLoader();
  var matText = new THREE.MeshBasicMaterial({ color: 0x444444 });

  fontLoader.load("/static/fonts/Arial.json",function(fnt){ 

      var params = {size: 5, height: 0.5,  curveSegments: 6,  font: fnt,}

      var geoText = new THREE.TextGeometry(xSize.toString(), params);

      var  textZ = new THREE.Mesh(geoText, matText);
      geoText.computeBoundingSphere();
      textZ.translateZ(-50 - 2 *  geoText.boundingSphere.radius);
      textZ.translateX(-geoText.boundingSphere.radius);
      textZ.translateY(10);
      textZ.rotateX(-Math.PI / 2);

      geoText = new THREE.TextGeometry(zSize.toString(), params);
      var  textX = new THREE.Mesh(geoText, matText);
      geoText.computeBoundingSphere();
      textX.translateX(-50 - 2 *  geoText.boundingSphere.radius);
      textX.translateZ(-geoText.boundingSphere.radius);
      textX.translateY(10);
      
      textX.rotateX(-Math.PI / 2);
      textX.rotateZ(-Math.PI / 2);

      geoText = new THREE.TextGeometry(bldg_height.toString(), params);
      geoText.computeBoundingSphere();

      var  textY = new THREE.Mesh(geoText, matText);
      textY.translateX(-50 - geoText.boundingSphere.radius);
      textY.translateZ(-50  - geoText.boundingSphere.radius);
      textY.translateY(buffer + bldg_height  + 1.5  * geoText.boundingSphere.radius  );
      geoText.computeBoundingSphere();
      textY.rotateY(-Math.PI / 4);

      var axes = new THREE.Object3D();

      console.log(axes)
      axes.name = "axes";
      axes.add(xAxis);
      axes.add(yAxis);
      axes.add(zAxis);
      axes.add(textZ);
      axes.add(textX);
      axes.add(textY);

      scene.add(axes);
  })

}
function generateInteriorModel(data) {

//rotating for rhino axes
// data.geometry.rotateX( Math.PI / 2); 

// Instantiate an exporter
var exporter = new STLExporter();
// Parse the input and generate the ply output
dloadInterior = exporter.parse( data );

// data.geometry.rotateX( -Math.PI / 2); 

}

function generateExteriorModel(data) {

//rotating for rhino axes
data.geometry.rotateX( Math.PI / 2); 

  // Instantiate an exporter
var exporter = new STLExporter();
// Parse the input and generate the ply output
dloadExterior = exporter.parse( data );

data.geometry.rotateX( -Math.PI / 2); 

}

//https://ourcodeworld.com/articles/read/189/how-to-create-a-file-and-generate-a-download-with-javascript-in-the-browser-without-a-server

function download(filename, text) {
  var element = document.createElement('a');
  element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
  element.setAttribute('download', filename);
  element.style.display = 'none';
  document.body.appendChild(element);

  element.click();
  document.body.removeChild(element);
}

function downloadCanvas(filename){

  var thisImage = new Image();
  thisImage = document.getElementById("c").toDataURL("image/png");
  
  var element = document.createElement('a');
  element.setAttribute('download', filename);
  element.setAttribute('href', thisImage);
  element.style.display = 'none';
  document.body.appendChild(element);

  element.click();
  document.body.removeChild(element);
}


document.querySelector("#downloadInterior").onclick = function()
  {
    // console.log(dloadInterior)
    download("interior_" + active_model.slice(-17, -4) + ".stl", dloadInterior)
  }  

document.querySelector("#downloadExterior").onclick = function()
{
  // console.log(dloadExterior)
  download("exterior_" + active_model.slice(-17, -4) + ".stl", dloadExterior)
}  

document.querySelector("#downloadSnapshot").onclick = function()
{
  // console.log(dloadExterior)
  downloadCanvas("snap_" + active_model.slice(-17, -4) + ".png");
}  



document.querySelector("#context").onchange = function()
{
  let toggle = $('#context').prop('checked')
  toggleContext(toggle)
}  

window.onload = (event) => {
  let toggle = $('#context').prop('checked')
  toggleContext(toggle)
};

document.querySelector("#site").onchange = function()
{
  let select_site = $('#site').prop('value')

  if(document.getElementById("context").checked == true)
  {
    console.log("removing context...")
    let objToRemove = scene.getObjectByName("context");
    scene.remove(objToRemove)
    document.getElementById("context").checked = false
  }

  if(select_site == "Canary Wharf")
  {
    fn_site = "/static/models/inputs/context_cw.obj";
    document.getElementById("site_image").src = "static/img/site_footprint_cw.png";
  }
  else if (select_site == "St Mary Axe")
  {
    fn_site = "/static/models/inputs/context_sm.obj"
    document.getElementById("site_image").src = "static/img/site_footprint_sm.png";
  }
}  