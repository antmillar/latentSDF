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

app = Flask(__name__ , static_url_path = '/static' )

#prevent file caching
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

#relative paths
cwd = os.getcwd()
model_path = cwd + '/static/models/torch/'
input_path = cwd + '/static/models/inputs'
output_path = cwd + '/static/models/outputs'

#globals/initial values
Details = namedtuple("Details", ["Floors", "Taper", "FloorRotation", "MaxCoverage", "MinCoverage"])
Bounds = namedtuple('Bounds', ['xMin', 'xMax', 'yMin', 'yMax'])
latent_bounds = Bounds(-1.5, 1.0, -1.0, 1.0)

active_model = ''
torch_model =  cwd + '/static/models/torch/default.pth'
height = 100
coverage = ""
check_site_boundary = ""
latent_loaded = False
contours = False
floors = False
show_context = "false"
model_details = Details(0,0,0,0,0)
latents = np.empty([0])
annotations = np.empty([0])
distances = np.empty([0])
points = ""
titles = np.empty([0])

#ROUTES--------------------

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

#upload torch model route
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():

        return render_template('main.html', latent_bounds = list(latent_bounds),  active_model = active_model, height = height, coverage = coverage, contours = contours, floors = floors, model_details = model_details, annotations = annotations, show_context=show_context)
        
#main route
@app.route('/main',  methods = ["GET", "POST"])
def main():

    global latent_bounds, latent_loaded, torch_model, annotations, latents, titles

    #on load create latent image 
    if(not latent_loaded):

        latent_space.updateLatent(latent_bounds, torch_model, latents)
        latent_loaded = True

    if request.method == 'POST': 

        #HANDLES FILE UPLOADING
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

                #add 1-indexed column for jinja
                annotations = np.c_[np.array([i for i in range(1, annotations.shape[0] + 1)]), annotations]

                #update bounds
                minX = round(min(val[0] for val in latents),1)
                maxX = round(max(val[0] for val in latents),1)
                minY = round(min(val[1] for val in latents),1)
                maxY = round(max(val[1] for val in latents),1)
                latent_bounds = Bounds(minX, maxX, minY, maxY)

                #update latent space
                latent_space.updateLatent(latent_bounds, torch_model, latents)

            else:

                print("Invalid File Type")
 
        #HANDLES 3D MODEL GENERATION
        if(request.form.get("generateSlices")):

            #delete old models
            files = glob.glob(output_path + "/*")
            for f in files:
            
                os.remove(f)
                print(f"deleted - {f}")

            #get data from POST
            global height, show_context, points
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

            points = request.form.get("pathPoints")

            slice_vectors = request.form.get("slices")
            slice_vectors = slice_vectors.split(",")

            coords = np.zeros((len(slice_vectors)//2, 2))
            coords[:,0] = slice_vectors[::2]
            coords[:,1] = slice_vectors[1::2]

            torch.cuda.empty_cache()

            #generate the 3D Model
            global contours, floors, model_details
            model_name, contours, floors, model_details =  eval.generate_model(coords, height, taper, rotation, torch_model)

            global active_model
            active_model = '/static/models/outputs/' + model_name

        #HANDLES UPDATING LATENT SPACE
        if(request.form.get("xMin")):

            latent_bounds = Bounds(float(request.form.get("xMin")), float(request.form.get("xMax")), float(request.form.get("yMin")), float(request.form.get("yMax")))
            latent_space.updateLatent(latent_bounds, torch_model, latents)

        #HANDLES UPDATING COVERAGE
        if(request.form.get("scoverage") == ""):

            if(request.form.get("ssite")):

                print("building site extents heatmap...")

                site_name = request.form.get("ssite_name")

                check_site_boundary = request.form.get("ssite")

                latent_space.updateLatent(latent_bounds, torch_model, latents, site_name, coverage_threshold = False, check_site_boundary = check_site_boundary)
            else:

                global coverage
                coverage = ""
                latent_space.updateLatent(latent_bounds, torch_model, latents, coverage_threshold = False) 
        
        if(request.form.get("scoverage")):

            print("building coverage heatmap...")

            coverage = request.form.get("scoverage")

            try:
                coverage = float(coverage)

            except:
                coverage = ""

            latent_space.updateLatent(latent_bounds, torch_model, latents, coverage_threshold =  coverage, check_site_boundary = False)



    return render_template('main.html', latent_bounds = list(latent_bounds), active_model = active_model, height = height, points=points, coverage = coverage, contours = contours, floors = floors, model_details = model_details, annotations = annotations, show_context=show_context)


if __name__ == '__main__':
    # app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000))) //use this version of line if running inside docker
    app.run(debug=True)