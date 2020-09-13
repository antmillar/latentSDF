import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy
from PIL import Image
import mcubes
import matplotlib.pyplot as plt
import time
from skimage import measure
from collections import namedtuple
import random

#internal imports
from .architectures import deepSDFCodedShape
from .utils import *
from .primitives import Box

#dirs
cwd = os.getcwd()
dir_image = cwd + '/static/img'
dir_data = cwd + '/static/models'
dir_output = cwd + '/static/models/outputs'
model_path = cwd + '/static/models/torch'

#static
batch_size = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
res = 50
ptsSample = np.float_([[x, y] 
                for y in  np.linspace(50, -50, res) 
                for x in np.linspace(-50, 50, res)])
pts = torch.Tensor(ptsSample).to(device)



def generate_slices(model, slice_vectors, rotation):

    start_rotation = 0
    sliceCount = len(slice_vectors)
    slices = []
    coverages = []
    endLayer = torch.ones(res, res)

    #add closing slice to first layer for marching cubes later
    slices.append(endLayer)

    for vector in slice_vectors[:sliceCount]:

      start_rotation += rotation
      rot_mat = create_rotation_matrix(start_rotation) 

      #rotate the sdf input pts
      pts_rotated = torch.mm(pts, rot_mat)

      sdf, coverage = process_slices(model, pts_rotated, torch.tensor(vector))
      coverages.append(coverage)



      slices.append(sdf.detach().cpu())

    return slices, coverages

def process_slices(model, pts, latent):

    sdf = model.forward(latent, pts)

    coverage = get_area_covered(sdf)
    # x = np.pad(x, pad_width=1, mode='constant', constant_values=0)

    sdf = sdf.view(res, res).detach().cpu().numpy()

    #pad the outermost edge of the SDF with positive vals to prevent issues with marching cubes
    sdf_padded = np.pad(sdf[1:res - 1, 1 : res - 1], pad_width=1, mode='constant', constant_values=1.0)

    return torch.tensor(sdf_padded), coverage


def taper_slices(slices, slices_to_taper):

  #get the deepest point in layers to taper
  #SHOULD CHanGE TO MAX??
  deepest_pt = min([torch.min(sl) for sl in slices[-slices_to_taper:]]) 
  
  print("deepest point to taper : ", deepest_pt)

  #tapering
  for i in range(slices_to_taper):
    #sq asymptote
    shrinkage = 1.0 / (slices_to_taper * slices_to_taper ) * -1 * deepest_pt
    slices[-slices_to_taper + i] = slices[-slices_to_taper + i].detach().cpu() + shrinkage * i * i

  return slices


@funcTimer
def generateModel(slice_vectors, height, taper, rotation, model_path):

    print("generating 3d model...")
    numSlices = height
    slice_count = len(slice_vectors)
 
    print("loading model...")
    model = load_torch_model(model_path)

    slices = []
    coverages = []
    cores = []
    endLayer = torch.ones(res, res)
    start_rotation = 0
   
    #update slices with rotation, and calculate coverages
    print("generating SDFs")
    slices, coverages = generate_slices(model, slice_vectors, rotation)

    print("converting 2D slices to 3D SDF...")
    gridDistanceWidth = 4.0 #because the max inside and outside is 1 and -1, 4 = 1 + 1 - - 1 - - 1
    cellSize = gridDistanceWidth / res

    #updates the values in a 3D SDF after stacking multiple 2D SDFs, not a perfect approach but certainly improvement over nothing.
    for myIndex, mySlice in enumerate(slices):
      for otherIndex, otherSlice in enumerate(slices):

        #rescaling because performing sqrt later on small numbers otherwise underflow
        rescale = 1000
        baseSlice = mySlice * rescale

        s = otherSlice * rescale
        # print(f"layers to search{torch.min(s) / cellSize}")
        dist = abs(otherIndex - myIndex)

        distancesViaNewLayer = np.square(s) + np.power(dist * rescale * cellSize, 2) 
        insideMask = mySlice < 0.0
        outsideMask = np.invert(insideMask)
        #compare with euclidean square dist via new layer
        diff = np.multiply(baseSlice,baseSlice) - distancesViaNewLayer

        #if diff greater than zero potential replacement
        mask = diff > 0
        mask2 = diff <= 0

        diffsToKeep = np.multiply(mask,  distancesViaNewLayer)
        diffsToKeep = np.multiply(diffsToKeep, insideMask)
        # print(torch.sum(diffsToKeep > 0))
        diffsToKeep = np.sqrt(diffsToKeep)
        diffsToKeep /= rescale


        diffsToKeep2 = np.multiply(mask2,  distancesViaNewLayer)
        diffsToKeep2 = np.multiply(diffsToKeep2, outsideMask)
        diffsToKeep2 = np.sqrt(diffsToKeep2)
        diffsToKeep2 /= rescale
        
        diffsmask = diffsToKeep == 0
        diffsmask2 = diffsToKeep2 == 0

        finalmask = diffsmask == diffsmask2

        mySlice = diffsToKeep + diffsToKeep2 + np.multiply(finalmask, mySlice)
        # print(np.unique(mySlice))

    print("tapering slices...")
    slices_to_taper = int(slice_count * taper / 100.0) * (slice_count // height)
    slices = taper_slices(slices, slices_to_taper)

    #add closing slice to last layer
    slices.append(endLayer)

    #stack slices into np array
    stacked = np.stack(slices)

    #generate the isocontours
    contours = []
    floors = []
    floor_labels = []
    samples = 400
    floor_samples = 40
    contour_every = 3
    floor_every = 1
    level = 0.0

    floor_height = 3
    floor_every = (slice_count // height) * floor_height
    slice_height = 1 / (slice_count // height) 

    print("generating floors...")
  #ADDING FLOORS
    ##funny indexing due to extra ends added to close form
    for idx, s in enumerate(slices[1:-slices_to_taper - 1:floor_every]):
        
        # level = -idx * taper/len(slices)

        #after first point and checking previous layer not empty
        if(idx > 0 and len(floors[idx - 1]) > 0):
            start_point = floors[idx - 1][0][0][:2] #previous layer first coordinate
            # slice_contours = extractContours(s, samples, level)
            floor_contours, labels = extractContours(s, floor_samples, level, start_point)

        else:
            floor_contours, labels = extractContours(s, floor_samples, level)

        a = []
        for floor in floor_contours:


            test = np.c_[floor, idx * floor_height * np.ones((floor.shape[0], 1))]
            # test[:,2] += 1 #subtract one for the extra base plane
            # test[:,2]*= floor_height #need to scale up the height
            test[:,0] -= 25.0
            test[:,1] -= 25.0
            test = test.tolist()
    
            a.append(test)# add extra column for height

        floors.append(a)
        floor_labels.append(labels)


    print("generating contours...")
    #create vertical contours
    ySlices = []
    for i in range(stacked.shape[1]):
      ySlices.append(stacked[:,i,:])


    for idx, s in enumerate(ySlices[::contour_every]):
        
        #after first point and checking previous layer not empty
        if(idx > 0 and len(contours[idx - 1]) > 0):
            start_point = contours[idx - 1][0][0][:2] #previous layer first coordinate
            # slice_contours = extractContours(s, samples, level)
            slice_contours, _ = extractContours(s, samples, level, start_point)

        else:
            slice_contours, _ = extractContours(s, samples, level)

        a = []
        for contour in slice_contours:

            #swap this column as we are slicing vertically now
              
              test = np.c_[contour, 1 * idx * contour_every * np.ones((contour.shape[0], 1))]

              test[:,[0, 2]] = test[:,[2, 0]]
              test[:,2] -= 1 #move down a floor

              test[:,2]*= slice_height #need to scale up the height
              #centering
              test[:,0] -= 25.0
              test[:,1] -= 25.0 
              test = test.tolist()
              a.append(test)

        contours.append(a)


    lblCounts = [len(np.unique(lbl)) for lbl in floor_labels]

    maxlbls  = max(lblCounts)

    lvlstocheck = [i for i, x in enumerate(lblCounts) if x == maxlbls]


    def most_frequent_label(List): 
        labels = set(List)
        labels.remove(0) #don't want to include outside space
        from collections import Counter
        # print(Counter(List))
        return max(labels, key = List.count) 
      
    #if maxlbls level, find most frequent element and how many there are

    #get most frequent contour label
    most_freq = [most_frequent_label(floor_labels[level].flatten().tolist()) for level in lvlstocheck]
    print(most_freq)

    #get how many of these there are
    occurences = [np.sum(floor_labels[level] == a) for level, a in zip(lvlstocheck, most_freq)]
    print(occurences)

    #find the level with the smallest biggest contour
    levelwithsmallestbiggestcontour = lvlstocheck[np.argmin(occurences)]
    print(levelwithsmallestbiggestcontour)

    #mask using this label and get value of lowest element in contour
    mask = floor_labels[levelwithsmallestbiggestcontour] == most_freq[lvlstocheck.index(levelwithsmallestbiggestcontour)]
   
    #get just the negative values
    vals = np.multiply(slices[levelwithsmallestbiggestcontour + 1].detach().numpy(), mask)
    print(np.min(vals)) 
    
    vals = vals.transpose()


    def get_vertical_outliers(x, y):
      '''
      Checks how many points lie outside of an sdf in this vertical cell
      '''

      axis = [s.numpy().transpose()[x, y] for s in slices[1:-1]]

      #count the number of positives in the core central axis
      positives = len([val for val in axis if val >= 0])

      print("points outside surface : ", positives)

      return positives



    def find_core_center(vals):

      #get the negative values
      potentialVals = list(np.unique(vals))

      #ensure boundary not in choices
      potentialVals.remove(0)

      num_samples = min(10, len(potentialVals))
      valsChosen = random.sample(potentialVals, num_samples)
      valsChosen.append(np.min(vals))

      print("random selection" , valsChosen)
      pos = []

      for choice in valsChosen:
        c = np.where(vals == choice)
        x = c[0][0]
        y = c[1][0]

        pos.append(get_vertical_outliers(x, y))


      minima = np.where(np.array(pos) == min(pos))[0]
      print("minima : ", minima)

      best_val = min([valsChosen[idx] for idx in minima])
      print("best val", best_val)


      #find in index of lowest element in contour

      c = np.where(vals == np.min(vals))
      print("core center using min sdf value : ", np.array([c[1][0], c[0][0]]))

      c = np.where(vals == best_val)
      print("core center after optimising with random search : ", np.array([c[1][0], c[0][0]]))

      core_center = np.array([c[1][0], c[0][0]])
      return core_center


    core_diameter = 5

    #get all contours for the level selected
    core_centers = []

    # get number of contours in this floor
    level_contours = np.unique(floor_labels[levelwithsmallestbiggestcontour])

    for i in range(1, len(level_contours) ):

      mask = floor_labels[levelwithsmallestbiggestcontour] == i

      vals = np.multiply(slices[levelwithsmallestbiggestcontour + 1].detach().numpy(), mask)
      print(np.min(vals)) 
      
      vals = vals.transpose()

      core_center = find_core_center(vals)

      square = Box(core_diameter, core_diameter, core_center)

      cores  = [square.field.reshape(res, res) for i in range(numSlices - slices_to_taper)]


      core_centers.append(cores)

    
    intersection = np.minimum.reduce([core for core in core_centers])
    cores = intersection


    #filter out empty contours/floors

    # floors = [item for item in floors if item != []]
    contours = [item for item in contours if item != []]

    #get the 0 iso surface for outer facade
    verts, faces = mcubes.marching_cubes(stacked, 0)

    #get iso surface for inner core/s
    stackedCores = np.stack(cores)
    vertsCore, facesCore = mcubes.marching_cubes(stackedCores, 0)

    #normalize verts to 50
    verts[:,1:] /= (res / 50)
    vertsCore[:,1:] /= (res / 50)

    # print(verts.shape)
    verts[:,0] -= 1
    verts[:,0] *= slice_height

    # translate to threejs space
    verts[:,1] -= 25.0
    verts[:,2] -= 25.0

    vertsCore[:,1] -= 25.0 #translate to threejs space
    vertsCore[:,2] -= 25.0


    new_arr = np.ones((faces.shape[0], faces.shape[1] + 1), dtype=np.int32) * 3
    new_arr[:,1:] = faces

    # Export the result
    timestr = time.strftime("%d%m%y-%H%M%S")
    fn = "model-" + timestr + ".obj"
    core_fn = "core_" + fn

    mcubes.export_obj(verts, faces,  os.path.join(dir_output, fn))
    mcubes.export_obj(vertsCore, facesCore,  os.path.join(dir_output, core_fn))

    # model details for app
     
    minCoverage = np.round(min(coverages) * 100.0, 2)
    maxCoverage = np.round(max(coverages) * 100.0, 2)

    Details = namedtuple("Details", ["Floors", "Taper", "Rotation", "MaxCoverage", "MinCoverage"])
    model_details = Details(numSlices, taper, rotation, maxCoverage, minCoverage)

    return fn, contours, floors, model_details


def getIndex(contours, point):

  # print(np.linalg.norm(np.(contours-point), axis = -1))
  idx = np.linalg.norm(np.abs(contours-point), axis = -1).argmin()

  return idx


def extractContours(s, num_samples, level = 0.0, start_point = None):

  # Find contours at a constant value of 0.0
  contours = measure.find_contours(s, level)
  s_binary = s < 0
  labels = measure.label(s_binary)
  startIndex = 0 

    
  #extract number of points required
  sampled_contours = []

  if(len(contours) == 0):
      return sampled_contours, labels

  contour_samples = num_samples#//len(contours)

  for contour in contours:


    if(type(start_point) is np.ndarray):
      startIndex = getIndex(contour, start_point)

    if(contour_samples > len(contour)):
        stepSize = 1
    else:
        stepSize = len(contour) // contour_samples

    contour = np.vstack((contour[startIndex:], contour[:startIndex])) #reorganise list to start at nearest pt
    sampled_contour = contour[::stepSize]
    # sampled_contour = sampled_contour[:contour_samples ] #bit hacky
    sampled_contour = np.round(sampled_contour, 0)

    sampled_contours.append(sampled_contour)

  return sampled_contours, labels

