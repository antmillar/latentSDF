#python library imports
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import torch
import json
import random
import threading
import time
import numpy as np
from collections import namedtuple

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
load_path = cwd + '/static/models/load'
input_path = cwd + '/static/models/inputs'
output_path = cwd + '/static/models/outputs'
mesh_path = cwd + '/static/models/meshes'

Bounds = namedtuple('Bounds', ['xMin', 'xMax', 'yMin', 'yMax'])
latent_bounds = Bounds(-1.5, 1.0, -1.0, 1.0)
img_source  = '/static/img/latent_grid.png'
img_source_hm  = '/static/img/coverage_heatmap.png'

active_model = ''
height = 100
coverage = False
latent_loaded = False
contours = False
floors = False
model_details = False

##TODO

#autoload latent at start with defaults
#seed designs align with vectors, store them in dict? pass from python
#show distance between seed designs somehow?
#display the model in GUI?
#dropdown  with active model

#make the seed latent vectors dynamic/store them
#in latent space capture details of the grid 




#base route
@app.route('/')
def index():
    return render_template('index.html')

#model route
@app.route('/modelView')
def modelView():
    return render_template('modelView.html')

#base route
@app.route('/main',  methods = ["GET", "POST"])
def main():

    #on load create latent image 
    global latent_bounds, latent_loaded
    if(not latent_loaded):
        
        latent_space.updateLatent(latent_bounds)
        latent_loaded = True
        # return redirect(url_for('main'))

    if request.method == "POST":

        #generate 3d model
        if(request.form.get("generateSlices")):

            global height
            height = int(request.form.get("modelHeight"))

            try:
                taper = float(request.form.get("modelTaper"))
                
            except:
                taper = 0.0

            sliceVectors = request.form.get("slices")
            sliceVectors = sliceVectors.split(",")
            coords = np.zeros((len(sliceVectors)//2, 2))
            coords[:,0] = sliceVectors[::2]
            coords[:,1] = sliceVectors[1::2]

            torch.cuda.empty_cache()
            global contours, floors, model_details
            model_name, contours, floors, model_details =  eval.evaluate(coords, height, taper)
            global active_model
            active_model = '/static/models/outputs/' + model_name

        #update the latent space
        if(request.form.get("updateLatent")):

            latent_bounds = Bounds(float(request.form.get("xMin")), float(request.form.get("xMax")), float(request.form.get("yMin")), float(request.form.get("yMax")))
            latent_space.updateLatent(latent_bounds)

        #update the latent space
        if(request.form.get("scoverage")):

            cov = request.form.get("scoverage")
            try:
                cov = float(cov)
            except:
                cov = False
            print(cov)

            latent_space.updateLatent(latent_bounds, cov)
            

    return render_template('main.html', latent_bounds = list(latent_bounds), img_source = img_source, img_source_hm = img_source_hm, active_model = active_model, height = height, coverage = coverage, contours = contours, floors = floors, model_details = model_details )

# #route to hold the latest status 
# @app.route('/progress/<int:thread_id>')
# def progress(thread_id):
#     global statusText
#     return str(statusText)

# #handles uploading files via https forms into python
# @app.route('/upload', methods = ['GET', 'POST'])
# def upload_file():

#     global statusText
#     if request.method == 'POST':

#         f = request.files['file']

#         #upload file from local and save to folder, downsample if too dense for model
#         if(f.filename[-4:].lower() == ".ply"):

#             fileToCopy = secure_filename(f.filename)

#             f.save(os.path.join(load_path, fileToCopy))

#             print(load_path + "/" + fileToCopy)

#             statusText = "subsampling cloud.."
#             pcd = preprocess_utils.down_sample(load_path, fileToCopy, statusText)

#             statusText = "removing outliers and normalizing..."
#             preprocess_utils.standardise(pcd, fileToCopy, input_path)

#             statusText = "view mode"

#         else:

#             print("invalid file format")

    
#     return redirect(url_for('modelViewer'))




# @app.route('/modelViewer' , methods=["GET", "POST"])
# def modelViewer():

#     global statusText

#     inputFiles = os.listdir(input_path)    
#     outputFiles = os.listdir(output_path)    
#     meshFiles = os.listdir(mesh_path)  

#     if request.method == "POST":


#         # removes files from app folder
#         if(request.form.get("fileNameRemove")):

#             fileToRemove = request.form.get("fileNameRemove")

#             print(input_path + "/" + fileToRemove)

#             statusText = f"removing file : {fileToRemove}.."

#             os.remove(input_path + "/" + fileToRemove)

#             try:
#                 os.remove(output_path + "/" + fileToRemove[:-4] + "_labels.ply")
#             except:
#                 pass

#             try:
#                 os.remove(mesh_path + "/" + fileToRemove[:-4] + ".obj")
#             except:
#                 pass

#             try:
#                 os.remove(mesh_path + "/" + fileToRemove[:-4] + ".mtl")
#             except:
#                 pass

#             statusText = "view mode"

#         #runs model
#         elif(request.form.get("fileNameInput")):
        
#             fileName = request.form.get("fileNameInput")

#             torch.cuda.empty_cache()

#             print(input_path + "/" + fileName)

#             statusText = "calculating stats.."

#             #get stats for cleaned pointcloud
#             _, _, _, _, _, density = preprocess_utils.get_stats(input_path, fileName)
            
#             print(f"density : {density}")

#             statusText = "evaluating model.."
#             #load filtered pointcloud, apply the model and save to new ply file
#             eval.evaluate(input_path, fileName, density)
        
#         #creates mesh
#         elif (request.form.get("fileNameOutput")):

#             statusText = "reconstructing mesh.."
        
#             fileName = request.form.get("fileNameOutput")
#             filterList = []

#             #get the label filters
#             if (request.form.get("filters")):
#                 filterList = request.form.get("filters").split(",")
#             print(filterList)

#             reconstruction.save_to_mesh(input_path, output_path, mesh_path, fileName, filters = filterList)

#             statusText = "view mode"

#         #downloads mesh
#         elif (request.form.get("fileNameDownload")):

#             fileName = request.form.get("fileNameDownload")[:-4] + ".obj" 

#             print(f"downloading {fileName}")

#             #open3d seems to always create an MTL file, can't figure out how to remove it. But the model should load fine without the MTL.
#             return send_from_directory(directory=mesh_path, filename=fileName , as_attachment=True)

#         #https://en.wikipedia.org/wiki/Post/Redirect/Get
#         return redirect(url_for('modelViewer'))

#     return render_template('modelViewer.html', inputFiles = inputFiles , outputFiles = outputFiles, meshFiles = meshFiles)


if __name__ == '__main__':
    # app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000))) //use this version of line if running inside docker
    app.run(debug=True)
