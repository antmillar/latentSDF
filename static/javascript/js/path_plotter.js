// Initialise globals
var path = new Path();
var nodes = [];
var history = [];
var texts = [];

// JavaScript Interop
globals.getSlices = extractSlices

//defaults
width = 600
margin = 50;
path.strokeColor =  new Color(1,0.0, 0.0, 0.5);
path.strokeWidth = 2;

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

            console.log(slices)
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
    console.log(slices)
    globals.slices = slices
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
        var node = new Path.Circle(event.point, 5);
        node.strokeWidth = 5;
        
        nodes.push(node);

        //format all nodes
        for (i = 0; i < nodes.length; i++) {
            nodes[i].strokeColor = new Color(1.0, 0.55, 0.25, 0.5);
            } 

        //update end nodes colors to different
        nodes[0].strokeColor =  new Color(0.0,0.5, 0.25, 0.8);
        nodes.slice(-1)[0].strokeColor = new Color(1.0, 0.0, 0.0, 0.8);

        //update list of points selected
        var hist = new PointText(new Point(width + margin + margin/10.0, 2 * margin + history.length * 20));
        hist.fontSize = 6;
        hist.fontFamily = "sans-serif";
        hist.fillColor = 'black';
        
        var coords = mouseToLatent(event.point);
        hist.content = "[ " + coords.x.toFixed(2) + " , " + coords.y.toFixed(2) + " ]";
        history.push(hist);

        //update the path
        path.add(event.point);
        path.smooth();

        //update the slice list on each click
        extractSlices(globals.slice_count * globals.height, nodes);
    }
    
    //on right click remove segment
    if(event.event.which == 2)
    {
        path.removeSegment(path.segments.length - 1);

        var nodeToRemove = nodes.pop();
        nodeToRemove.remove();

        var histToRemove = history.pop();
        histToRemove.remove();

    }
}