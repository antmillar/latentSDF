#python library imports
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import glob
import torch
import json
import random
import threading
import time
import numpy as np
from collections import namedtuple
import zipfile

# #local module imports
import python_modules.deepsdf.eval as eval
import python_modules.deepsdf.latent_space as latent_space
# import python_modules.preprocess.utils as preprocess_utils
# import python_modules.postprocess.reconstruction as reconstruction

#show the status text top right
global statusText
statusText = "view mode"

app = Flask(__name__ , static_url_path = '/static' )

 #prevent file caching
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

#relative paths
cwd = os.getcwd()
output_path = cwd + '/static/img/output'
model_path = cwd + '/static/models/torch/'
input_path = cwd + '/static/models/inputs'
output_path = cwd + '/static/models/outputs'
mesh_path = cwd + '/static/models/meshes'

Details = namedtuple("Details", ["Floors", "Taper", "FloorRotation", "MaxCoverage", "MinCoverage"])

Bounds = namedtuple('Bounds', ['xMin', 'xMax', 'yMin', 'yMax'])
latent_bounds = Bounds(-1.5, 1.0, -1.0, 1.0)
img_source  = '/static/img/latent_grid.png'
img_source_hm  = '/static/img/coverage_heatmap.png'

active_model = ''
torch_model =  '/home/anthony/repos/latentSDF/static/models/torch/default.pth'
height = 50
coverage = ""
latent_loaded = False
contours = False
floors = False
show_context = "false"
model_details = Details(0,0,0,0,0)
latents = np.empty([0])
annotations = np.empty([0])
distances = np.empty([0])

titles = np.empty([0])

##TODO
#tidy up code in python 
#think about global variables
#tidy up JS
#how to convert to ship app?
#create landing page
#3d latent space
# bug after the model is loaded and then restarted
#reset distance explorer at beginning


#base route
@app.route('/')
def index():
    return render_template('index.html')

#design analysis route
@app.route('/analysis',  methods = ['GET', 'POST'])
def analysis():

    if(latents != []):
        latent_space.prepare_analysis(torch_model, latents)

    if request.method == 'POST': 
 
        if(request.form.get("seedDesign")):
            index = int(request.form.get('seedDesign'))
            global distances
            distances = latent_space.calculate_distances(index, annotations)

    return render_template('analysis.html', latents = latents, annotations = annotations, titles = titles ,distances=distances)

#model route
@app.route('/modelView')
def modelView():
    return render_template('modelView.html')

#upload torch model route
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():



        return render_template('main.html', latent_bounds = list(latent_bounds), img_source = img_source, img_source_hm = img_source_hm, active_model = active_model, height = height, coverage = coverage, contours = contours, floors = floors, model_details = model_details, annotations = annotations, show_context=show_context)
        
        # return redirect(url_for('main', latent_bounds = list(latent_bounds), img_source = img_source, img_source_hm = img_source_hm, active_model = active_model, height = height, coverage = coverage, contours = contours, floors = floors, model_details = model_details, annotations = annotations, show_context=show_context))


@app.route('/downloader', methods = ['GET'])
def download_file():

    print(f"downloading {cwd + active_model}")

    #open3d seems to always create an MTL file, can't figure out how to remove it. But the model should load fine without the MTL.
    print(active_model)
    return send_from_directory(directory=output_path, filename=active_model[-23:] , as_attachment=True) #HACKAKCKAKCAKCKAKC

#base route
@app.route('/main',  methods = ["GET", "POST"])
def main():

    global latent_bounds, latent_loaded, torch_model, annotations, latents, titles

    #on load create latent image 
    if(not latent_loaded):

        #delete old models on restart
        files = glob.glob(output_path + "/*")
        for f in files:
            
            os.remove(f)
            print(f"deleted - {f}")

        latent_space.updateLatent(latent_bounds, torch_model, latents)
        latent_loaded = True

    if request.method == 'POST': 

        #if file uploaded
        if(request.files):
            print("uploading file...")

            f = request.files['file']

            #load zip file and extract latents and model
            if(f.filename[-4:].lower() == ".zip"):

                fileToCopy = secure_filename(f.filename)
                fn = os.path.join(model_path, fileToCopy)
                f.save(fn)
                print(f"file : {f.filename} uploaded successfully")

                with zipfile.ZipFile(fn, 'r') as zip_ref:
                    zip_ref.extractall(model_path)
                
                #load the model from the zip
                torch_model = model_path + "model.pth"

                #load the latents from the zip
                float_formatter = "{:.2f}".format
                np.set_printoptions(formatter={'float_kind':float_formatter})
                annotations = np.load(model_path + "seeds.npy").astype(object) #object type as need to hold lists not just values

                titles = annotations[0, :]
                annotations = annotations[1:, :]

                #convert string latents to lists
                latents = [json.loads(latent) for latent, *_ in annotations]
                annotations[:,0] = latents 

                #sort by x coord
                sortedOrder = np.array(latents)[:,0].argsort()
                annotations[0:,:] = annotations[0:,:][sortedOrder]

                #add 1-indexed column
                annotations = np.c_[np.array([i for i in range(1, annotations.shape[0] + 1)]), annotations]

                #update bounds
                minX = min(val[0] for val in latents)
                maxX = max(val[0] for val in latents)
                minY = min(val[1] for val in latents)
                maxY = max(val[1] for val in latents)
                
                latent_bounds = Bounds(minX, maxX, minY, maxY)
                latent_space.updateLatent(latent_bounds, torch_model, latents)

            else:

                print("Invalid File Type")
 
        #generate 3d model
        if(request.form.get("generateSlices")):

            global height, show_context
            height = int(request.form.get("modelHeight"))

            try:
                taper = float(request.form.get("modelTaper"))
                
            except:
                taper = 0

            try:
                rotation = float(request.form.get("modelRotation"))
                
            except:
                rotation = 0

            show_context = request.form.get("show_context")

            slice_vectors = request.form.get("slices")
            slice_vectors = slice_vectors.split(",")

            coords = np.zeros((len(slice_vectors)//2, 2))
            coords[:,0] = slice_vectors[::2]
            coords[:,1] = slice_vectors[1::2]

            torch.cuda.empty_cache()

            global contours, floors, model_details
            model_name, contours, floors, model_details =  eval.generateModel(coords, height, taper, rotation, torch_model)

            global active_model
            active_model = '/static/models/outputs/' + model_name

        #update the latent space
        if(request.form.get("xMin")):

            latent_bounds = Bounds(float(request.form.get("xMin")), float(request.form.get("xMax")), float(request.form.get("yMin")), float(request.form.get("yMax")))
            latent_space.updateLatent(latent_bounds, torch_model, latents)

        #update the latent space
        if(request.form.get("scoverage") == ""):

            global coverage
            coverage = ""
            latent_space.updateLatent(latent_bounds, torch_model, latents, coverage)

        if(request.form.get("scoverage")):

            coverage = request.form.get("scoverage")

            try:
                coverage = float(coverage)

            except:
                coverage = ""

            latent_space.updateLatent(latent_bounds, torch_model, latents, coverage)


    return render_template('main.html', latent_bounds = list(latent_bounds), img_source = img_source, img_source_hm = img_source_hm, active_model = active_model, height = height, coverage = coverage, contours = contours, floors = floors, model_details = model_details, annotations = annotations, show_context=show_context)


if __name__ == '__main__':
    # app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000))) //use this version of line if running inside docker
    app.run(debug=True)
