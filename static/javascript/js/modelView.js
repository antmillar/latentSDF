import * as THREE from 'https://threejsfundamentals.org/threejs/resources/threejs/r113/build/three.module.js';
import {OrbitControls} from 'https://threejsfundamentals.org/threejs/resources/threejs/r113/examples/jsm/controls/OrbitControls.js';
import {OBJLoader2} from 'https://threejsfundamentals.org/threejs/resources/threejs/r113/examples/jsm/loaders/OBJLoader2.js';
import {OBJExporter} from 'https://threejsfundamentals.org/threejs/resources/threejs/r113/examples/jsm/exporters/OBJExporter.js';


var scene, camera, renderer, canvas;

canvas = document.querySelector('#c');


  //scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xBBAAAA);

  // //grid
  // const size = 100;
  // const divisions = 50;
  // let helper = new THREE.GridHelper( size, divisions, 0x444444, 0x444444 );
  // scene.add( helper );

  //camera
  const fov = 45;
  const aspect = 1;  // the canvas default
  const clipNear = 0.01;
  const clipFar = 5000;
  const radius = 30
  camera = new THREE.PerspectiveCamera(fov, aspect, clipNear, clipFar);
  camera.position.set(radius * Math.cos( 3 * Math.PI/4) , 30, radius * Math.sin( 3 * Math.PI/4));
  camera.lookAt( 0, 0, 0 );

  //mouse controls
  const controls = new OrbitControls(camera, canvas);
  controls.target.set(0, 5, 0);
  controls.update();

  // //lighting
  // var hemiLight = new THREE.HemisphereLight( 0xffffff, 0.5);
  // scene.add( hemiLight );

  var ambient = new THREE.AmbientLight( 0xffffff );
  scene.add( ambient );
        

  var dirLight = new THREE.DirectionalLight( 0xffffff, 0.5);
  dirLight.position.set( -20, 20, 20 );
  dirLight.castShadow = true;
  dirLight.shadow.mapSize = new THREE.Vector2(1024, 1024);
  scene.add(dirLight);

  // var dirLight2 = new THREE.DirectionalLight( 0xffffff, 0.5);
  // dirLight2.position.set( 15, 30, 10 );
  // dirLight2.castShadow = true;
  // dirLight2.shadow.mapSize = new THREE.Vector2(1024, 1024);
  // scene.add(dirLight2);

  renderer = new THREE.WebGLRenderer({antialias:true, canvas : canvas});
  let width = 800
  let height = 800
  renderer.setSize(width, height)
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  if(active_model == "") 
  {
    
  }
  else
  {

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
    console.log("core model : " , model)
    model.material = matCore;

    model.scale.set(0.1,0.1,0.1);
    model.rotation.z = Math.PI / 2; 

    scene.add(model);

  })

  // LOAD THE BUILDING, FLOORS and CONTOURS
  let loader = new OBJLoader2();
  loader.load(active_model, function(obj){

    //materials

    let matGlass = new THREE.MeshPhysicalMaterial( {
      // color: 0xA4CBD4,
      color: 0xAAAAAA,
      opacity: 0.25,
      // side: THREE.DoubleSide,
      transparent: true,
      // reflectivity: 0.5,
    } );

    let matGlass2 = new THREE.MeshPhysicalMaterial( {
      // color: 0xA4CBD4,
      color: 0x555555,
      opacity: 0.5,
      side: THREE.DoubleSide,
      transparent: true,
      // depthTest: false,

    } );

    let matFloor = new THREE.MeshPhongMaterial( {color: 0xAA9999, side: THREE.DoubleSide} );
    let matContours = new THREE.LineBasicMaterial( { color: 0xEEEEEE, linewidth: 1.5,   } );

    //geometry
    let model = obj.children[0];
    console.log("bldg model : " , model)

    model.geometry.computeVertexNormals();
    model.material = matGlass2
    model.castShadow = true;

    let plane = new THREE.PlaneGeometry( 20, 20, 0 );
    let ground = new THREE.Mesh( plane, matFloor );
    ground.receiveShadow = true;

    //scaling/rotating
    model.scale.set(0.1,0.1,0.1);
    model.rotation.z = Math.PI / 2; 

    ground.rotation.x = -Math.PI / 2; 


    //create floors in building
    for(let i = 0; i < floors.length; i++)
    {
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
        let floor = new THREE.Mesh( geometryFloor, matGlass );
        floor.castShadow = true;
        floor.scale.set(0.1,0.1,0.1);
        floor.rotation.x = Math.PI / 2; 
        floor.rotation.y = Math.PI; 
        floor.translateZ(points[0].z / 10.0);
  
        edge.scale.set(0.1,0.1,0.1);
        edge.rotation.x = Math.PI / 2; 
        edge.rotation.y = Math.PI; 
        edge.translateZ(points[0].z / 10.0);
  
        scene.add( floor, edge );
      }
    }

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
      contourLine.scale.set(0.1,0.1,0.1);
      contourLine.rotation.x = Math.PI / 2; 
      contourLine.rotation.y = Math.PI; 

      // contourLine.updateMatrix();

      // geometryContour.applyMatrix( contourLine.matrix );

      scene.add(contourLine)

      generate(contourLine)
    }
    scene.add(model,  ground);



  });
  
}

animate();
// download(scene) 

function animate() {

  renderer.render(scene,camera);
  requestAnimationFrame(animate);

}

var dload

function generate(da) {

  // Instantiate an exporter
var exporter = new OBJExporter();
// Parse the input and generate the ply output
var data = exporter.parse( da );

dload = data

}


function download(filename, text) {
  var element = document.createElement('a');
  element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
  element.setAttribute('download', filename);

  element.style.display = 'none';
  document.body.appendChild(element);

  element.click();

  document.body.removeChild(element);
}


document.querySelector("#download").onclick = function()
  {
    console.log(dload)
    download("test.obj", dload)

  }  
