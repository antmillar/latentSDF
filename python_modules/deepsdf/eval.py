#various functions taken from https://github.com/daveredrum/Pointnet2.ScanNet/blob/master/visualize.py
#other functions added
 
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from .architectures import deepSDFCodedShape
from.utils import funcTimer

import json
from pathlib import Path
import numpy
from PIL import Image
import mcubes
import matplotlib.pyplot as plt
import time
from skimage import measure
import openmesh as om
import pyvista as pv

#static
cwd = os.getcwd()
dir_image = cwd + '/static/img'
dir_model = cwd + '/torchModels'
dir_data = cwd + '/static/models'
dir_output = cwd + '/static/models/outputs'
batch_size = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

res = 50
ptsSample = np.float_([[x, y] 
                for y in  np.linspace(-50, 50, res) 
                for x in np.linspace(-50, 50, res)])
pts = torch.Tensor(ptsSample).to(device)

#load a torch model and generate slices
def evaluate(sliceVectors, height):
 
    print("loading model...")
    model_path = os.path.join(dir_model, "8floorplans.pth")
    model_path = os.path.join(dir_model, "8floorplans_selflearn.pth")


    model = deepSDFCodedShape()#.cuda()
    model.load_state_dict(torch.load(model_path, map_location=device))
    # res = 50
    latent = torch.tensor( [-2, 1]).to(device)
    # print("labelling points...")
    # pred = predict_label(model, dataloader)
    
    # print("saving PLY file...")
    # save_to_PLY(fn, pred)
    numSlices = 100

    return generateModel(sliceVectors, model, height)



def forward(model, coords, feats):
    pred = []

    coord_chunk, feat_chunk = torch.split(coords.squeeze(0), batch_size, 0), torch.split(feats.squeeze(0),batch_size, 0)
    assert len(coord_chunk) == len(feat_chunk)
    for coord, feat in zip(coord_chunk, feat_chunk):
        output = model(torch.cat([coord, feat], dim=2))
        pred.append(output)

    pred = torch.cat(pred, dim=0) # (CK, N, C)
    outputs = pred.max(2)[1]

    return outputs

#remove points duplicated in the dataset random sampling
def filter_points(coords, pred):
    assert coords.shape[0] == pred.shape[0]
    print(f"pre filter point count : {coords.shape[0]}")

    #hash the xyz coords as a string
    coord_hash = [hash(str(coords[point_idx][0]) + str(coords[point_idx][1]) + str(coords[point_idx][2])) for point_idx in range(coords.shape[0])]

    #remove dups
    _, coord_ids = np.unique(np.array(coord_hash), return_index=True)

    #filter
    coord_filtered, pred_filtered = coords[coord_ids], pred[coord_ids]
    
    filtered = []

    for point_idx in range(coord_filtered.shape[0]):
        filtered.append(
            [
                coord_filtered[point_idx][0],
                coord_filtered[point_idx][1],
                coord_filtered[point_idx][2],
                labelMap[pred_filtered[point_idx]][0],
                labelMap[pred_filtered[point_idx]][1],
                labelMap[pred_filtered[point_idx]][2]
            ]
        )

    print(f"filtered point count : {len(filtered)}")
    return np.array(filtered)


def predict_label(model, dataloader : DataLoader):
    output_coords, output_pred = [], []
    print("predicting labels...")
    count = 0


    for data in dataloader:
        # unpack

        coords, feats, targets, weights, _ = data
        coords, feats, targets, weights = coords.cuda(), feats.cuda(), targets.cuda(), weights.cuda()

        # feed
        pred = forward(model, coords, feats)

        # dump
        coords = coords.squeeze(0).view(-1, 3).cpu().numpy()
        pred = pred.view(-1).cpu().numpy()
        output_coords.append(coords)
        output_pred.append(pred)
        count+=1

    print("filtering points...")
    output_coords = np.concatenate(output_coords, axis=0)
    output_pred = np.concatenate(output_pred, axis=0)

    filtered = filter_points(output_coords, output_pred)
   
    return filtered

def save_to_PLY(fn : str, pred):

    #convert pred np array to list of tuples
    points = list(map(tuple, pred))

    points = np.array(
        points,
        dtype=[
            ("x", np.dtype("float32")), 
            ("y", np.dtype("float32")), 
            ("z", np.dtype("float32")),
            ("red", np.dtype("uint8")),
            ("green", np.dtype("uint8")),
            ("blue", np.dtype("uint8"))
        ]
    )

    #convert to ply elements
    plyData = PlyElement.describe(points, "vertex")
    plyData = PlyData([plyData])

    #save labelled model 
    output_fn = Path(fn).stem + "_labels.ply"
    plyData.write(os.path.join(dir_output, output_fn))
    print("saved as " + output_fn)



@funcTimer
def generateModel(sliceVectors, model, numSlices):

    print("generating 3d model...")

    def createSlice(pts, latent):

        sdf = model.forward(latent, pts)

        #need to generate a 50 x 50 sdf for the core

        circle = Circle(np.array([25, 25]), 20)
        # print(np.unique(circle.field))

        # intersection = np.maximum(circle.field * -1, sdf.detach().numpy())
        # print(type(intersection))
        # print(intersection.shape)
        return sdf.view(res, res)
        # return torch.tensor(intersection).view(res, res)

    slices = []


    endLayer = torch.ones(res, res)
    #add closing slice to first layer 
    slices.append(endLayer)

    for vector in sliceVectors:
        pixels = createSlice(pts, torch.tensor(vector))
        ##need to correct for values changing as slices are added in 3d
        slices.append(pixels.detach().cpu())

    #add closing slice to last layer
    slices.append(endLayer)
 
    ground_floor = slices[1]
    ground_floor_min = torch.min(ground_floor)
    ground_floor_min_coords = torch.nonzero(ground_floor == torch.min(ground_floor))
    
    stacked = np.stack(slices)
    core_diameter = 10
    core_center = np.array([25, 25])
    circle = Circle(np.array([res//2, res//2]), core_diameter)
    square = Box(core_diameter, core_diameter, core_center)
    core = [52 * square.field.reshape(res, res)]
    core = np.stack(core)

    intersection = np.maximum(core * -1, stacked)
    # stacked = intersection

    #generate the isocontours
    contours = []
    floor_height = 2
    samples = 40
    taper = 0
    contour_every = 4

    for idx, s in enumerate(slices[1::contour_every]):
        
        level = -idx * taper/len(slices)

        #after first point and checking previous layer not empty
        if(idx > 0 and len(contours[idx - 1]) > 0):
            start_point = contours[idx - 1][0][0][:2] #previous layer first coordinate
            # slice_contours = extractContours(s, samples, level)
            slice_contours = extractContours(s, samples, level, start_point)

        else:
            slice_contours = extractContours(s, samples, level)

        a = []
        for contour in slice_contours:
            a.append(np.c_[contour, floor_height * idx * contour_every * np.ones((contour.shape[0], 1))].tolist() )# add extra column for height

        contours.append(a)

    newIdx = len(contours)
    ySlices = []
    for i in range(stacked.shape[1]):
      ySlices.append(stacked[:,i,:])


    for idx, s in enumerate(ySlices[1::contour_every*2]):
        
        level = -idx * taper/len(ySlices)

        #after first point and checking previous layer not empty
        if(idx > 0 and len(contours[newIdx + idx - 1]) > 0):
            start_point = contours[newIdx + idx - 1][0][0][:2] #previous layer first coordinate
            # slice_contours = extractContours(s, samples, level)
            slice_contours = extractContours(s, samples, level, start_point)

        else:
            slice_contours = extractContours(s, samples, level)

        a = []
        for contour in slice_contours:

            #swap this column as we are slicing vertically now
              
              test = np.c_[contour, 1 * idx * contour_every*2 * np.ones((contour.shape[0], 1))]

              test[:,[0, 2]] = test[:,[2, 0]]
              test[:,2]*= floor_height #need to scale up the height
              test = test.tolist()
              a.append(test)

        contours.append(a)



    newIdx = len(contours)
    xSlices = []
    for i in range(stacked.shape[2]):
      xSlices.append(stacked[:,:,i])


    for idx, s in enumerate(xSlices[1::contour_every]):
        
        level = -idx * taper/len(xSlices)

        #after first point and checking previous layer not empty
        if(idx > 0 and len(contours[newIdx + idx - 1]) > 0):
            start_point = contours[newIdx + idx - 1][0][0][:2] #previous layer first coordinate
            # slice_contours = extractContours(s, samples, level)
            slice_contours = extractContours(s, samples, level, start_point)

        else:
            slice_contours = extractContours(s, samples, level)

        a = []
        for contour in slice_contours:

            #swap this column as we are slicing vertically now
              
              test = np.c_[contour, 1 * idx * contour_every * np.ones((contour.shape[0], 1))]


              test[:,[0, 2]] = test[:,[2, 0]]
              test[:,[0, 1]] = test[:,[1, 0]]
              test[:,2]*= floor_height #need to scale up the height
              test = test.tolist()
              a.append(test)

        contours.append(a)



    #filter out empty contours
    contours = [item for item in contours if item != []]


    #convert contours to JSON to be loaded in threeJS
    # contour = contours[1][0].tolist()
    # contour = json.dumps(data)

    # with open(os.path.join(dir_output, 'data.json'), 'w') as f:
    #   json.dump(data, f)

    # mesh = om.PolyMesh()

    # for idx in range(len(contours) - 1):

    #     for idx2, contour in enumerate(contours[idx]):

    #         handles1 = []
    #         handles2 = []

    #         next_layer_contours = (contours[idx + 1][i] for i in range(len(contours[idx + 1])))

    #         #something to fix here when tapering
    #         combined = np.vstack(next_layer_contours)


    #         #in the case where the model splits into more than one contour need to pick the order to pair up the points
    #         if(len(contours[idx + 1]) > len(contours[idx]) ):

    #             #merge the two new slices into one array, need to make this handle more
    #             # combined = np.vstack((contours[idx + 1][idx2], contours[idx + 1][idx2 + 1]))
    #             previousSlice = contours[idx][idx2]

    #         #find nearest point in new layer and add to handles
    #             for coord in combined:

    #                 #select item from first slice
    #                 vh = mesh.add_vertex(coord)
    #                 handles2.append(vh)
                    
    #                 #select closest item from second slice and remove
    #                 i = getIndex(previousSlice, coord)
    #                 previousSlice, coord2 = poprow(previousSlice, i)
    #                 vh2 = mesh.add_vertex(coord2)
    #                 handles1.append(vh2)

    #         else:

    #             #check if next layer has less contours
    #             # if(idx2 < len(contours[idx + 1])):
    #                 # nextSlice = contours[idx + 1][idx2]
    #             nextSlice = combined

    #             for coord in contour:
    #                 vh = mesh.add_vertex(coord)
    #                 handles1.append(vh)

    #                 i = getIndex(nextSlice, coord)
    #                 nextSlice, coord2 = poprow(nextSlice, i)
    #                 vh2 = mesh.add_vertex(coord2)
    #                 handles2.append(vh2)

    #             # for coord in contours[idx + 1][idx2]:
    #             #     vh = mesh.add_vertex(coord)
    #             #     handles2.append(vh)

    #             # #deal with case where the next level has a different number of contours


    #         for a in range(len(handles1)):
    #           mesh.add_face(handles1[a % len(handles1)], handles1[(a+1) % len(handles1)],  handles2[(a + 1) % len(handles2)], handles2[a % len(handles2)])

    # #close ends


    # for contour in contours[0]:
    #     handles1 = []
    #     for coord in contour:
    #         vh = mesh.add_vertex(coord)
    #         handles1.append(vh)

    #     mesh.add_face(handles1)


    # for contour in contours[len(contours) - 1]:
    #     handles1 = []
    #     for coord in contour:
    #         vh = mesh.add_vertex(coord)
    #         handles1.append(vh)
    #     mesh.add_face(handles1)

    ##flattens an irregular nested list
    import collections
    def flatten(l):
      for el in l:
        ##stop recursion once reach the coordinate list
          if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)) and not(len(el) == 3 and isinstance(el[0], float)):
              yield from flatten(el)
          else:
              yield el


    # flat_list = [item for sublist in contours for item in sublist]
    # flat_list = list(flatten(contours))
    # print(len(flat_list))
    # new_list = [item for sublist in contours for item in sublist]
    # list2 = [x for x in flat_list if  any(isinstance(item, float) for item in x)]
    # print(type(flat_list[0][0]))
    # print(flat_list[0])


    # print(contours.shape)
    # stacked = np.stack(contours)
    # om.write_mesh(os.path.join(dir_output, fn), mesh)

    # #get the 0 iso surface
    verts, faces = mcubes.marching_cubes(stacked, 0)


    # #normalize verts to 50
    verts[:,1:] /= (res / 50)
    # print(verts.shape)
    verts[:,0] *= floor_height


    # new_arr = np.ones((faces.shape[0], faces.shape[1] + 1), dtype=np.int32) * 3
    # new_arr[:,1:] = faces

    # verts = np.round(verts, 0)
    # mesh = pv.PolyData(verts, new_arr)
    # points = pv.PolyData(flat_list)
    # surf = points.delaunay_3d().extract_geometry().clean()
    # mesh = mesh.decimate(0.3)

    # print(verts % 1 == 0)
    # fn = "model-" + timestr + ".obj"
    # # Export the result
    timestr = time.strftime("%d%m%y-%H%M%S")

    fn = "model-" + timestr + ".obj"
    mcubes.export_obj(verts, faces,  os.path.join(dir_output, fn))
    # pv.save_meshio(os.path.join(dir_output, fn), surf)

    return fn, contours


def poprow(my_array,pr):
    """ row popping in numpy arrays
    Input: my_array - NumPy array, pr: row index to pop out
    Output: [new_array,popped_row] """
    i = pr
    pop = my_array[i]
    new_array = np.vstack((my_array[:i],my_array[i+1:]))
    return new_array,pop

def getIndex(contours, point):

  # print(np.linalg.norm(np.(contours-point), axis = -1))
  idx = np.linalg.norm(np.abs(contours-point), axis = -1).argmin()

  return idx


def extractContours(s, num_samples, level = 0.0, start_point = None):

  # Find contours at a constant value of 0.0
  contours = measure.find_contours(s, level)
  startIndex = 0 


  # contour_list = [contour for contour in contours]

  # contours = np.vstack(contour_list)
    
  #extract number of points required
  sampled_contours = []

  if(len(contours) == 0):
      return sampled_contours

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

  return sampled_contours



class Shape():

  def __init__(self):
    self.res = 50
    self.generateField(0, 50, self.res)

  def generateField(self, start, end, steps):

    '''Generates the a 2D signed distance field for a shape
    
            Parameters:
                    start (int): Start value of coordinates
                    end (int): End value of coordinates
                    steps (int): Number of steps in each dimension

            Returns:
                    outputField (list): List of signed distance field values
    '''
    self.start = start
    self.end = end
    self.steps = steps
    self.pts = np.float_([[x, y] 
                    for y in  np.linspace(start, end, steps) 
                    for x in np.linspace(start, end, steps)])

    self.field = np.float_(list(map(self.sdf, self.pts))).reshape(steps*steps, 1)


    # self.normalizeField()

    return True

  def normalizeField(self):

    '''Normalizes the signed distance field to be within [-1,1]'''

    absMin = abs(np.min(self.field))
    absMax = abs(np.max(self.field))

    absAbsMax = max(absMin, absMax)

    self.field /= absAbsMax
      

#subclasses

class Circle(Shape):

  def __init__(self, center : np.array, radius : float):


    self.center = center
    self.radius = radius / 2
    super().__init__()

  def sdf(self, p):

    return np.linalg.norm(p - self.center) - self.radius

class Box(Shape):

  def __init__(self, height: float, width: float, center: np.array):

    if(height <= 0 or width <= 0):
      raise ValueError("Height or Width cannot be negative")

    self.hw = np.float_((height / 2.0, width / 2.0))
    self.center = center
    super().__init__()

  def sdf(self, p):

    #translation
    p = p - self.center
    
    d = abs(p) - self.hw

    return np.linalg.norm([max(0, item) for item in d]) + min(max(d[0], d[1]), 0)
