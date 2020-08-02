// Create a Paper.js Path to draw a line into it:
var path = new Path();
var nodes = [];
var history = [];
var texts = [];

// JavaScript Interop


// function internalClicked() {
//     var slices = extractSlices(100);
//     console.log(slices);
// }

width = 600

path.strokeColor =  new Color(1,0.0, 0.0, 0.5);

path.strokeWidth = 2;

var raster = new Raster({
    source: '/static/img/latent_grid.png',
    position: view.center
});

raster.insertBelow(path)

//https://stackoverflow.com/questions/47136708/how-to-resize-a-rasterimage-in-paper-js
Raster.prototype.rescale = function(width, height) {
    this.scale(width / this.width, height / this.height);
};

raster.rescale(width, width);
margin = 100;

var text = new PointText(new Point(350, 30));
text.fontSize = 16;
text.fontFamily = "sans-serif";
text.fillColor = 'black';

// var annotate = new PointText(new Point(margin/2, 135));
// annotate.fontSize = 30;
// annotate.fontFamily = "sans-serif";
// annotate.fillColor = 'black';
// annotate.content = parseFloat(globals.latentBounds[3]);

function addAxisLabels(){

    var len = globals.latentBounds.length;
    var positions = [new PointText(new Point(margin + 30, width + margin + margin /2  )) ,
                     new PointText(new Point(width + margin - margin/2, width + margin + margin /2  )),
                     new PointText(new Point(margin/2, margin + width - 25)),
                     new PointText(new Point(margin/2, margin + margin / 2))]

    for (var i = 0; i < len; i++) {
        var annotate = positions[i]
        annotate.fontSize = 30;
        annotate.fontFamily = "sans-serif";
        annotate.fillColor = 'black';
        annotate.content = parseFloat(globals.latentBounds[i]);
    }

}

addAxisLabels()
// Set the content of the text item:

function mouseToLatent(point){

    var pos = point - new Point(margin, margin)
    pos = pos / width;

    xMin = parseFloat(globals.latentBounds[0]);
    xMax = parseFloat(globals.latentBounds[1]);
    yMin = parseFloat(globals.latentBounds[2]);
    yMax = parseFloat(globals.latentBounds[3]);

    var xVal = xMin + pos.x * (xMax - xMin);
    var yVal = yMax - pos.y * (yMax - yMin);

    //rescale by 10% due so images are centered on coordinates, 1.1 because 10 images. need to fix this
    latentPoint = new Point(xVal, yVal);

    return latentPoint
}
//annotates cursors position on the latent space
raster.onMouseMove = function(event) {

    var latentPoint = mouseToLatent(event.point);

    text.content = "[ " + latentPoint.x.toFixed(2) + " , " + latentPoint.y.toFixed(2) + " ]";
}

//hides cursors position when not in latent space
raster.onMouseLeave = function(event) {

    text.content = "";
}

function extractSlices(sliceCount){

    //check if there are actually any nodes!
    slices = [];

    //deal with edge case of a single point, which is not a valid path yet
    if(path.length == 0)
    {
        for(i = 0; i <sliceCount; i++)
        {
            var pt = path.segments[0].point / width;
            slices.push([pt.x.toFixed(2), pt.y.toFixed(2)]);
         }
    }

    else
    {
        var offsetStep = path.length / sliceCount;

        for(i = 0; i <sliceCount; i++)
        {
            var point = path.getPointAt(offsetStep * i);
            var pt = point / width;
            slices.push([pt.x.toFixed(2), pt.y.toFixed(2)]);
        }
    }
    // console.log(slices)
    return slices;
}


function onMouseDown(event) {

    //on left click
    if(event.event.which == "1")
    {
        
        var node = new Path.Circle(event.point, 5);
        // node.insertAbove(raster);
        node.strokeWidth = 5;
        
        nodes.push(node);


        var hist = new PointText(new Point(710, 200 + history.length * 20));
        hist.fontSize = 10;
        hist.fontFamily = "sans-serif";
        hist.fillColor = 'black';

        var coords = mouseToLatent(event.point)
        hist.content = history.length + " : [ " + coords.x.toFixed(2) + " , " + coords.y.toFixed(2) + " ]"
        history.push(hist);

        for (i = 0; i < nodes.length; i++) {
            nodes[i].strokeColor = new Color(1.0, 0.55, 0.25, 0.5);
          } 


        nodes[0].strokeColor =  new Color(0.0,0.5, 0.25, 0.8);
        nodes.slice(-1)[0].strokeColor = new Color(1.0, 0.0, 0.0, 0.8);

        path.add(event.point)
        path.smooth() //need to contrain values on path within the range if want to use this.

        globals.slices = extractSlices(100);
    }
    
    //on right click
    if(event.event.which == 2)
    {
        path.removeSegment(path.segments.length - 1);

        var nodeToRemove = nodes.pop();
        nodeToRemove.remove();

        var histToRemove = history.pop();
        histToRemove.remove();

    }

}