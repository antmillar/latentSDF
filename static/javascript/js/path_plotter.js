// Initialise globals
var path = new Path();
var nodes = [];
var points = [];
var history = [];
var texts = [];
var arrows = [];
var lastArrows = []
var lastPath = new Path();

// JavaScript Interop
globals.getSlices = extractSlices





//defaults
width = 600
margin = 50;
path.strokeColor =  new Color(0.0,0.0, 0.0, 0.65);
path.strokeWidth = 1.5;
path.dashArray = [10, 4];
lastPath.strokeColor =  new Color(0.0,0.0, 0.0, 0.85);
lastPath.strokeWidth = 1.5;






//add background canvas image
var raster = new Raster({
    source: globals.latent_img_source,
    position: view.center
});

raster.insertBelow(path)

//https://stackoverflow.com/questions/47136708/how-to-resize-a-rasterimage-in-paper-js
Raster.prototype.rescale = function(width, height) {
    this.scale(width / this.width, height / this.height);
};

raster.rescale(width, width);

//add heatmap image
var rasterHM = new Raster({
    source: globals.heatmap_img_source,
    position: view.center
});

rasterHM.insertBelow(raster)
rasterHM.rescale(width, width);

var text = new PointText(new Point(width / 2 - margin, 30));
text.fontSize = 16;
text.fontFamily = "sans-serif";
text.fillColor = 'black';

addAxisLabels()




//recreate path
last_points = []
var point_arr = globals.points.split(',').map(parseFloat)
for(i = 0; i < point_arr.length; i += 2)
{
  var pt = new Point([point_arr[i],point_arr[i+1]]);
  last_points.push(pt);
  lastPath.add(pt);
  lastPath.smooth();
}
var n = new Path.Circle(last_points[0], 5);
n.strokeColor = new Color(0.0,0.5, 0.25, 0.8);

n.strokeWidth = 3;
var m = new Path.Circle(last_points.slice(-1)[0], 5);
m.strokeColor = new Color(1.0, 0.0, 0.0, 0.8);
m.strokeWidth = 3;


lastArrows = addArrows(lastArrows, last_points, lastPath)

//converts coordinate from mouse world space to latent space
function mouseToLatent(point){

    var pos = point - new Point(margin, margin)
    pos = pos / width;

    xMin = parseFloat(globals.latent_bounds[0]);
    xMax = parseFloat(globals.latent_bounds[1]);
    yMin = parseFloat(globals.latent_bounds[2]);
    yMax = parseFloat(globals.latent_bounds[3]);

    var xVal = xMin + pos.x * (xMax - xMin);
    var yVal = yMax - pos.y * (yMax - yMin);

    //rescale by 10% due so images are centered on coordinates, 1.1 because 10 images. need to fix this
    latentPoint = new Point(xVal, yVal);

    return latentPoint
}


//extracts list of slices from the current path
function extractSlices(sliceCount, nodes){

    //check if there are actually any nodes!
    slices = [];

    //deal with edge case of a single point, which is not a valid path yet
    if(path.length == 0.0)
    {
        for(i = 0; i <sliceCount; i++)
        {   
            pt = mouseToLatent(path.segments[0].point)
            slices.push([pt.x.toFixed(2), pt.y.toFixed(2)]);
         }
    }

    else
    {
        var offsetStep = path.length / sliceCount;

        //if discrete sample at the nodes only, not along whole path
        if(globals.discrete)
        {
            var stackSize =  Math.floor(sliceCount/ nodes.length);
            var remainder = sliceCount % stackSize;

            for(i = 0; i <nodes.length; i++)
            {
                var center = getCenter(nodes[i]);
                var pt = mouseToLatent(center)

                for(j = 0; j < stackSize; j++)
                {
                    slices.push([pt.x.toFixed(2), pt.y.toFixed(2)]);
                }
            }

            for(k = 0; k < remainder; k++)
            {
                slices.push([pt.x.toFixed(2), pt.y.toFixed(2)]);
            }

        }
        else
        {

            for(i = 0; i <sliceCount; i++)
            {
                var point = path.getPointAt(offsetStep * i);
                var pt = point / width;
                pt = mouseToLatent(point)
                slices.push([pt.x.toFixed(2), pt.y.toFixed(2)]);
            }
        }

    }

    new_points = [];

    for(i = 0; i < points.length; i++)
    {
        new_points.push(points[i].x, points[i].y)
    }

    globals.points = new_points;
    globals.slices = slices;
}

function getCenter(n){


    var a = n.segments[0].point;
    var b = n.segments[1].point;
    var c = n.segments[2].point;
    var d = n.segments[3].point;

    var centerX = (a.x + b.x + c.x + d.x) / 4.0;
    var centerY = (a.y + b.y + c.y + d.y) / 4.0;

    return new Point(centerX, centerY)
}


//adds the axis labels
function addAxisLabels(){

    var len = globals.latent_bounds.length;
    var positions = [new PointText(new Point(margin, width + margin + margin /2  )) ,
                     new PointText(new Point(width + margin, width + margin + margin /2  )),
                     new PointText(new Point(margin/2, margin + width)),
                     new PointText(new Point(margin/2, margin ))]

    for (var i = 0; i < len; i++) {
        var annotate = positions[i]
        annotate.fontSize = 20;
        annotate.fontFamily = "sans-serif";
        annotate.fillColor = 'black';
        annotate.content = parseFloat(globals.latent_bounds[i]);
    }

}

//HANDLES MOUSE EVENTS

//annotates cursors position on the latent space
raster.onMouseMove = function(event) {

    var latentPoint = mouseToLatent(event.point);

    text.content = "Latent : [ " + latentPoint.x.toFixed(2) + " , " + latentPoint.y.toFixed(2) + " ]";
}

//hides cursors position when not in latent space
raster.onMouseLeave = function(event) {

    text.content = "";
}

//handles left and right click events
function onMouseDown(event) {

    //on left click add node to path
    if(event.event.which == "1")
    {
        if(points.length > 0)
        {
            from = points[points.length-1]
        }

        points.push(event.point)

        var node = new Path.Circle(event.point, 5);

        nodes.push(node);

        //format all nodes
        for (i = 0; i < nodes.length; i++) {
            nodes[i].strokeColor = new Color(0.0, 0.0, 0.0, 0.0);
            } 

        //update end nodes colors to different
        nodes[0].strokeColor =  new Color(0.0,0.5, 0.25, 0.8);
        nodes[0].strokeWidth = 3;
        nodes.slice(-1)[0].strokeColor = new Color(1.0, 0.0, 0.0, 0.8);
        nodes.slice(-1)[0].strokeWidth = 3;

        //update list of points selected
        // var hist = new PointText(new Point(width + margin + margin/10.0, 2 * margin + history.length * 20));
        // hist.fontSize = 6;
        // hist.fontFamily = "sans-serif";
        // hist.fillColor = 'black';
        
        // var coords = mouseToLatent(event.point);
        // hist.content = "[ " + coords.x.toFixed(2) + " , " + coords.y.toFixed(2) + " ]";
        // history.push(hist);

        //update the path
        path.add(event.point);
        path.smooth();
    
        arrows = addArrows(arrows, points, path);
        //update the slice list on each click
        extractSlices(globals.slice_count * globals.height, nodes);
    }


    
    //on middle button click remove segment
    if(event.event.which == 2)
    {
        path.removeSegment(path.segments.length - 1);

        if(nodes.length > 0)
        {
            var nodeToRemove = nodes.pop();
            nodeToRemove.remove();
        }

        // var histToRemove = history.pop();
        // histToRemove.remove();

        if(points.length > 0)
        {
            var pointToRemove = points.pop();
        }

        if(arrows.length > 0)
        {
            var arrowToRemove = arrows.pop();
            arrowToRemove.remove();

            var arrowToRemove2 = arrows.pop();
            arrowToRemove2.remove();
        }
    }
}

function addArrows(arws, arr, pth)
{

    for(var i = 0; i < arws.length; i++)
    {
        arws[i].removeSegments();
    }

    arws = []

    for(var i = 1; i < arr.length; i++)
    {
        var offset = pth.getLocationOf(arr[i]).offset;

        var point = pth.getPointAt(offset);

        // Find the tangent vector at the given offset
        // and give it a length of 60:
        var tangent = pth.getTangentAt(offset) * 10;

        var line = new Path({
            segments: [point, point + tangent],
            strokeColor:  new Color(0.0,0.0, 0.0, 0.75),
            strokeWidth: 2,
        })
        var line2 = line.clone();

        line.rotate(30, line.getPointAt(line.length));
        line2.rotate(-30, line.getPointAt(line.length));
        arws.push(line)
        arws.push(line2)
    }
    return arws;
}


function downloadCanvas(filename){

    var thisImage = new Image();
    thisImage = document.getElementById("paperCanvas").toDataURL("image/png");
    
    var element = document.createElement('a');
    element.setAttribute('download', filename);
    element.setAttribute('href', thisImage);
    element.style.display = 'none';
    document.body.appendChild(element);

    element.click();
    document.body.removeChild(element);


    // document.getElementById("downloader").href = document.getElementById("canvas").toDataURL("image/png").replace(/^data:image\/[^;]/, 'data:application/octet-stream');
}

document.querySelector("#downloadPath").onclick = function()
{
    downloadCanvas("latent_path.png")
}  


//update discrete on/off
$('#discrete').on('change', function() {
window.globals.discrete  = $('#discrete').prop('checked')
console.log($('#discrete').prop('checked'))
extractSlices(globals.slice_count * globals.height, nodes);

});