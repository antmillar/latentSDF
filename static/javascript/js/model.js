//import loader helpers
import {OBJLoader2} from 'https://threejsfundamentals.org/threejs/resources/threejs/r113/examples/jsm/loaders/OBJLoader2.js';
import {PLYLoader} from 'https://threejsfundamentals.org/threejs/resources/threejs/r113/examples/jsm/loaders/PLYLoader.js';
import Shaders from './shaders.js'
import Scene from './scene.js'

export default class Model 
{

  constructor(name)
  {
    this.name = name;
    this.activeScene;
    this.scenes = {};

    this.eventLoaded = new Event('loaded');    
    this.pathInput = "/static/models/inputs/";
    this.pathLabelled = "/static/models/outputs/";
    this.pathMesh = "/static/models/meshes/";
    this.defaultPointSize = 2.0;
    this.defaultOpacity = 1.0;

    this.labelMap =   {

    "[152, 223, 138]":		 "floor",
    "[174, 199, 232]":		 "wall",
    "[31, 119, 180]" :		 "cabinet",
    "[255, 187, 120]":		 "bed",
    "[188, 189, 34]": 		 "chair",
    "[140, 86, 75]":  		 "sofa",
    "[255, 152, 150]":		 "table",
    "[214, 39, 40]":  		 "door",
    "[197, 176, 213]":		 "window",
    "[148, 103, 189]":		 "bookshelf",
    "[196, 156, 148]":		 "picture",
    "[23, 190, 207]":		 "counter",
    "[247, 182, 210]"	:	 "desk",
    "[219, 219, 141]":		 "curtain",
    "[255, 127, 14]":		 "refrigerator",
    "[227, 119, 194]":		 "bathtub",
    "[158, 218, 229]":		 "shower curtain",
    "[44, 160, 44]":  		 "toilet",
    "[112, 128, 144]":		 "sink",
    "[82, 84, 163]":      "other"

    }
  }


  //load the data from folders into scenes
  loadScenes(inputFiles, labelFiles, meshFiles)
  {
    var that = this;

    //load inputfiles into scenes
    inputFiles.forEach(function(item, index){

      var scene = new Scene(item);
      that.scenes[item] = scene;

    })

    for (const [name, scene] of Object.entries(that.scenes)) {

      //add files in input directory to scene object
      that.loadGeometry(name, scene, that.pathInput, "input");

      //add files in labelled directory to scene object
      let label_name = name.slice(0, -4) + "_labels.ply"
      if(labelFiles.includes(label_name))
        {
          that.loadGeometry(label_name, scene, that.pathLabelled, "output");
        }

      //add files in mesh directory to scene object
      let mesh_name = name.slice(0, -4) + ".obj"
      if(meshFiles.includes(mesh_name))
        {
          that.loadGeometry(mesh_name, scene, that.pathMesh, "mesh");
        }
      }

    console.log(that.scenes)
  }
  
  //load geometry from a fn
  loadGeometry(fn, scene, root, type)
  {
    let extn = fn.substr(fn.lastIndexOf('.') + 1);

    if(false)//fn in dest)
    {
        //pass;
    }
    else if(extn.toUpperCase() === "PLY")
    {
      this.loadPLY(fn, scene, root, type);
    }
    else if (extn.toUpperCase() === "OBJ" )
    {
      this.loadOBJ(fn, scene, root);
    }
    else if (extn.toUpperCase() === "MTL" )
    {
      console.log(`Skipped MTL File : ${fn}`);
    }
    else
    {
      console.error("Invalid Path Extension");
    }


  }

  //load PLY file from fn
  loadPLY(fn, scene, root, type){

    const plyLoader = new PLYLoader();

    plyLoader.load(root + fn, (geometry) => {

      console.log('Loading : ' + root + fn);

      geometry.computeVertexNormals();

      //get positions attribute
      let positions = geometry.getAttribute("position");
      let count = positions.count

      //assign point size attribute
      let pointSize = new Float32Array(count);
      pointSize.fill(this.defaultPointSize);
      geometry.setAttribute( 'pointSize', new THREE.BufferAttribute( pointSize, 1 ) );

      //assign opacity attribute
      let opacity = new Float32Array(count);
      opacity.fill(this.defaultOpacity);
      geometry.setAttribute( 'opacity', new THREE.BufferAttribute( opacity, 1.0 ) );

      //assign color attribute
      if(!geometry.getAttribute("color")){
        geometry.setAttribute( 'color', new THREE.BufferAttribute( new Float32Array( count * 3 ), 3 ) );
      }

      let colors =  geometry.getAttribute("color");
      let colorsScaled = colors.array.map(x => x * 255.0);

  

      //assign materials
      let shaderMaterial = new THREE.ShaderMaterial({

        vertexShader : Shaders.vertexShader(),
        fragmentShader : Shaders.fragmentShader(),
        // blending: THREE.AdditiveBlending,
        depthTest: false,
        transparent: true, //allows opacity
      })


      //create the point cloud
      let pcd = new THREE.Points( geometry, shaderMaterial );
      pcd.name = fn;
      pcd.labels = [];
      pcd.rotation.x = -Math.PI / 2; //align to threeJS
      // pcd.scale.set(2, 2, 2);

      //translate to origin
      var center = new THREE.Vector3();
      var size = new THREE.Vector3();
      var bbox = new THREE.Box3().setFromObject(pcd);

      bbox.getCenter(center);
      bbox.getSize(size);

      pcd.translateZ(center.y); //place on plane
      pcd.geometry.center();

      //get metadata
      pcd.ptCount = pcd.geometry.attributes.position.count;
      pcd.volume = (size.x * size.y * size.z);
      pcd.density = (pcd.ptCount / pcd.volume);


      //if file has labels process these
      if(type == "output"){


        //extract color r g b into triplets as strings e.g [""]
        var coords = colorsScaled.reduce(function(result, _, index, array) {

          if(index % 3 === 0)
          {
            result.push(array.slice(index, index + 3));
          }
          return result

        }, []);

        pcd.labelledPoints = coords.map(x=> this.labelMap[objToString(x)]);

        //extract the set of colors present
        let set = new Set(coords.map(JSON.stringify));
        let unique = Array.from(set).map(JSON.parse);

        pcd.labels = unique.map(x => this.labelMap[objToString(x)]);
        pcd.display = {};
        pcd.labels.forEach(label => {pcd.display[label] = true;});
        pcd.toggles = [];
      }


      //assign to scene objects
      if(type == "input")
      {      
        scene.inputPLY = pcd;
        scene.inputPLY.name = "inputPLY"
      }
      else if(type == "output")
      {
        scene.labelledPLY = pcd;
        scene.labelledPLY.name = "labelledPLY"

      }
  
      //dispatch event to say file loaded
      btnLoad.dispatchEvent(this.eventLoaded);
      console.log(`Loaded : ${fn}`);

     });
  }


  //load OBJ file from fn
  loadOBJ(fn, scene, root){
    {
      const objLoader = new OBJLoader2();

      console.log('Loading : ' + root + fn);
    
      objLoader.load(root + fn, (geometry) => {
    

        let mesh;

        //check if mesh nested
        if(!(mesh = geometry.children[0]))
        {

          mesh = geometry;
        }

        mesh.name = fn;
        mesh.rotation.x = -90 * Math.PI/180;

        //get mesh attributes
        let center = new THREE.Vector3();
        let bbox = new THREE.Box3().setFromObject(mesh);
        let size = new THREE.Vector3;

        bbox.getSize(size);
        bbox.getCenter(center);

        mesh.ptCount = mesh.geometry.attributes.position.count;
        mesh.volume = (size.x * size.y * size.z);
        mesh.density = (mesh.ptCount / mesh.volume);

        //move mesh to origin
        mesh.translateY(center.z);
        mesh.translateX(-center.x);
        
        var count = mesh.geometry.attributes.position.count;
        
        let scale = 1;

        mesh.scale.set(scale, scale, scale);

        var positions = mesh.geometry.getAttribute("position");

        console.log(positions.count);
      
        const mat = new THREE.MeshBasicMaterial( { vertexColors: THREE.VertexColors, side: THREE.DoubleSide } ); //color: 0xf1f1f1, 

        //needed if has mesh is nested in obj
        function initColor(parent, mtl) {
        parent.traverse((o) => {
            if (o.isMesh) {
                    o.material = mtl;
            }
          });
          }

        //assign vertex colors material to model
        initColor(mesh, mat);
      
        scene.mesh = mesh;
        scene.mesh.name = "mesh";

        //dispatch event to say file loaded
        console.log(`Loaded : ${fn}`);
        btnLoad.dispatchEvent(this.eventLoaded);
      });
    }
    }

  //pushs a file from javascript to python
  uploadFile()
  {
    console.log(document.querySelector('#uploadInput').files[0]);
    document.querySelector('#uploadFile').submit()
  }

  //running the model on input PLY file
  runModel()
  {

    //if there is an active scene
    if(this.activeScene)
    {

    document.querySelector("#fileNameInput").value = this.activeScene.name;
    document.querySelector('#btnModel').submit();

    } 
    else 
    {
      alert("Warning : No Scene Selected")
    }
  }

//tells python to create mesh from labelled point cloud
  createMesh()
  {

    if(this.activeScene){
      if(this.activeScene.labelledPLY){

        let filters = this.collateFilters();
        document.querySelector("#filters").value = filters;
        document.querySelector("#fileNameOutput").value = this.activeScene.name;
        document.querySelector('#btnMesh').submit();

      } 
      else 
      {
        alert("Warning : Must Generate a Labelled PLY File first by running the model")
      }
    }
    else
    {
      alert("Warning : No Scene Selected")
    }
    
  }

  //download generated mesh if available
  downloadMesh()
  {
    var that = this;
    if(this.activeScene){

      if(that.activeScene.mesh){
        document.querySelector("#fileNameDownload").value = that.activeScene.name;
        document.querySelector('#downloadFile').submit();
      }

      else

      {
        alert("Warning : No Mesh Created Yet")
      }
    }
    else
    {
      alert("Warning : No Model Selected")
    }
    
  }

  //delete a model from the app
  removeModel()
  {
    var that = this;
    // console.error("not implemented yet")
    if(this.activeScene){

      delete that.scenes[that.activeScene.name];
      console.log(that.scenes);

      let fileToRemove = that.activeScene.name;
      document.querySelector("#fileNameRemove").value = fileToRemove;
      document.querySelector('#removeFiles').submit()


      //dispatch event to trigger dropdown refresh
      btnLoad.dispatchEvent(this.eventLoaded);
      console.log(`Removed : ${that.activeScene.name}`);
    }
    else
    {
      alert("Warning : No Scene Selected")
    }
    
  }

  //generate list of labels to be filtered server side
  collateFilters()
  {
    var that = this;
    let labelsToFilter = [];

    Object.keys(that.activeScene.labelledPLY.display).forEach(function(item, index)
    {
      if(that.activeScene.labelledPLY.display[item] === false)
      {
        labelsToFilter.push(item)
      }
    })
    
    return labelsToFilter;
  }
}

//helper fn

//convert array to it's equivalent in a string
function objToString(obj)
{
  let arr = Object.values(obj);

  var str = "[";
  str += arr.join(", ");
  str += "]";
  
  return str;
}
