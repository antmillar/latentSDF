import * as THREE from 'https://threejsfundamentals.org/threejs/resources/threejs/r113/build/three.module.js';
import {OrbitControls} from 'https://threejsfundamentals.org/threejs/resources/threejs/r113/examples/jsm/controls/OrbitControls.js';
import {OBJLoader2} from 'https://threejsfundamentals.org/threejs/resources/threejs/r113/examples/jsm/loaders/OBJLoader2.js';

let scene, camera, renderer, canvas;

canvas = document.querySelector('#c');


  //scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x888888);

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

  //lighting
  var hemiLight = new THREE.HemisphereLight( 0xffffff, 0.5);
  scene.add( hemiLight );

  var dirLight = new THREE.DirectionalLight( 0xffffff, 1.0);
  dirLight.position.set( -10, 20, 10 );
  dirLight.castShadow = true;
  dirLight.shadow.mapSize = new THREE.Vector2(1024, 1024);
  scene.add(dirLight);

  var dirLight2 = new THREE.DirectionalLight( 0xffffff, 0.5);
  dirLight2.position.set( 15, 30, 10 );
  dirLight2.castShadow = true;
  dirLight2.shadow.mapSize = new THREE.Vector2(1024, 1024);
  scene.add(dirLight2);

  renderer = new THREE.WebGLRenderer({antialias:true, canvas : canvas});


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
      opacity:0.75,
      side: THREE.DoubleSide,
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
      color: 0x666666,
      opacity: 0.5,
      side: THREE.DoubleSide,
      transparent: true,
      // reflectivity: 0.5,
    } );

    let matGlass2 = new THREE.MeshPhysicalMaterial( {
      // color: 0xA4CBD4,
      color: 0x222222,
      opacity: 0.5,
      transparent: true,
    } );

    let matFloor = new THREE.MeshBasicMaterial( {color: 0x999999, side: THREE.DoubleSide} );
    let matContours = new THREE.LineBasicMaterial( { color: 0xEEEEEE, linewidth: 0.75} );

    //geometry
    let model = obj.children[0];
    console.log("bldg model : " , model)

    model.geometry.computeVertexNormals();
    model.material = matGlass2

    let plane = new THREE.PlaneGeometry( 5, 5, 0 );
    let ground = new THREE.Mesh( plane, matFloor );

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
        let matEdge = new THREE.LineBasicMaterial( { color: 0x222222, linewidth: 0.5 } )
        let edges = new THREE.EdgesGeometry( geometryFloor);
        let edge = new THREE.LineSegments( edges, matEdge );
        let floor = new THREE.Mesh( geometryFloor, matGlass );
  
        floor.scale.set(0.1,0.1,0.1);
        floor.rotation.x = Math.PI / 2; 
        floor.rotation.y = Math.PI; 
        floor.translateZ(points[0].z / 10.0);
  
        edge.scale.set(0.1,0.1,0.1);
        edge.rotation.x = Math.PI / 2; 
        edge.rotation.y = Math.PI; 
        edge.translateZ(points[0].z / 10.0);
  
        scene.add( model, edge );
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

      scene.add(contourLine)
    }

    scene.add(model, ground);
  });
  
}

animate();


function animate() {

  renderer.render(scene,camera);
  requestAnimationFrame(animate);

}
