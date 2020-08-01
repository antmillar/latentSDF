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
    source: '/static/img/latent.png',
    position: view.center,
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

// Set the content of the text item:


//annotates cursors position on the latent space
raster.onMouseMove = function(event) {

    var pos = event.point - new Point(margin, margin);
    pos /= width;
    text.content = "[ " + pos.x.toFixed(2) + " , " + pos.y.toFixed(2) + " ]"
}

//hides cursors position when not in latent space
raster.onMouseLeave = function(event) {

    text.content = "";
}

function extractSlices(sliceCount){

    //check if there are actually any nodes!

    slices = [];

    var offsetStep = path.length / sliceCount;

    for(i = 0; i <sliceCount; i++)
    {
        var point = path.getPointAt(offsetStep * i);
        var pt = point / width;
        slices.push([pt.x.toFixed(2), pt.y.toFixed(2)]);
    }

    return slices;
}

function onKeyDown(event) {

    if(event.key == 's')
    {
        var s = extractSlices(100);
        globals.slices = extractSlices(100);
        console.log(globals.slices)
    }
}

function onMouseDown(event) {

    //on left click
    if(event.event.which == "1")
    {
        
        var node = new Path.Circle(event.point, 5);
        // node.insertAbove(raster);
        node.strokeWidth = 5;
        
        nodes.push(node);


        var hist = new PointText(new Point(10, 200 + history.length * 20));
        hist.fontSize = 10;
        hist.fontFamily = "sans-serif";
        hist.fillColor = 'black';
        var p = event.point - new Point(margin, margin);
        var coords = p / width;
        hist.content = history.length + " : [ " + coords.x.toFixed(2) + " , " + coords.y.toFixed(2) + " ]"
        history.push(hist);

        for (i = 0; i < nodes.length; i++) {
            nodes[i].strokeColor = new Color(1.0, 0.55, 0.25, 0.5);
          } 


        nodes[0].strokeColor =  new Color(0.0,0.5, 0.25, 0.8);
        nodes.slice(-1)[0].strokeColor = new Color(1.0, 0.0, 0.0, 0.8);

        path.add(event.point)
        path.smooth() //need to contrain values on path within the range if want to use this.
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