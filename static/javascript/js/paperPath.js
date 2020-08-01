// Create a Paper.js Path to draw a line into it:
var path = new Path();
var nodes = [];
var history = [];
var texts = [];
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
        console.log(s);
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
        console.log(history.length)
        hist.fontSize = 10;
        hist.fontFamily = "sans-serif";
        hist.fillColor = 'black';
        console.log(event.point / width)
        var coords = event.point / width;
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

    // if(event.key == 'up')
    // {
    //     R += 0.01;
    //     text.content = 'R = ' + R.toPrecision(3);
    //     path.removeSegments();
    //     cobwebPath.removeSegments();
    //     identityPath.removeSegments();
    //     chaosPath.removeSegments();

    //     GraphLogisticMap(R);
    // }
    // if(event.key == 'down')
    // {
    //     path.add(MouseEven)
    //     R -= 0.01;
    //     text.content = 'R = ' + R.toPrecision(3);
    //     path.removeSegments();

    //     GraphLogisticMap(R);
    // }
}