import * as THREE from 'https://threejsfundamentals.org/threejs/resources/threejs/r113/build/three.module.js';
import {OrbitControls} from 'https://threejsfundamentals.org/threejs/resources/threejs/r113/examples/jsm/controls/OrbitControls.js';
import {OBJLoader2} from 'https://threejsfundamentals.org/threejs/resources/threejs/r113/examples/jsm/loaders/OBJLoader2.js';

let scene, camera, renderer, canvas;

canvas = document.querySelector('#c');

if(active_model == "")
{
  canvas.style.display="none";
}

else
{  
  //scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color("white");

  //grid
  const size = 100;
  const divisions = 50;
  let helper = new THREE.GridHelper( size, divisions, 0x444444, 0x444444 );
  scene.add( helper );

  //camera
  const fov = 45;
  const aspect = 1;  // the canvas default
  const clipNear = 0.01;
  const clipFar = 5000;
  const radius = 50
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
  dirLight.position.set( -5, 10, 5 );
  dirLight.castShadow = true;
  dirLight.shadow.mapSize = new THREE.Vector2(1024, 1024);
  scene.add(dirLight);

  var dirLight2 = new THREE.DirectionalLight( 0xffffff, 0.5);
  dirLight2.position.set( 5, 20, 10 );
  dirLight2.castShadow = true;
  dirLight2.shadow.mapSize = new THREE.Vector2(1024, 1024);
  scene.add(dirLight2);

  renderer = new THREE.WebGLRenderer({antialias:true, canvas : canvas});
  // renderer.setSize(window.innerWidth / 4,window.innerHeight / 4);
  // document.body.appendChild(renderer.domElement);

  let loader = new OBJLoader2();

  // console.log(window.globals.active_model)
  loader.load(active_model, function(obj){

    //materials

    let matGlass = new THREE.MeshPhysicalMaterial( {
      color: 0xA4CBD4,
      opacity: 0.75,
      side: THREE.DoubleSide,
      transparent: true,
      reflectivity: 0.5,
    } );

    let matFloor = new THREE.MeshBasicMaterial( {color: 0x444444, side: THREE.DoubleSide} );
    let matWireframe = new THREE.LineBasicMaterial( { color: 0x333333, linewidth: 1} );

    //geometry
    let model = obj.children[0];
    model.geometry.computeVertexNormals();
    model.material = matGlass

    let geometry = new THREE.EdgesGeometry( model.geometry );
    let wireframe = new THREE.LineSegments( geometry, matWireframe );
    let plane = new THREE.PlaneGeometry( 20, 20, 0 );
    let floor = new THREE.Mesh( plane, matFloor );

    //scaling/rotating
    model.scale.set(0.1,0.1,0.1);
    model.rotation.z = Math.PI / 2; 

    wireframe.scale.set(0.1,0.1,0.1);
    wireframe.rotation.z = Math.PI / 2; 

    floor.rotation.x = -Math.PI / 2; 

    //translating
    let center = new THREE.Vector3();
    let bbox = new THREE.Box3().setFromObject(model);

    bbox.getCenter(center);
    
    //move mesh to origin
    model.translateZ(-center.z);
    model.translateY(center.x);

    wireframe.translateZ(-center.z);
    wireframe.translateY(center.x);


    scene.add(model, wireframe, floor);
    animate();
  });


}
function animate() {

  renderer.render(scene,camera);
  requestAnimationFrame(animate);

}

