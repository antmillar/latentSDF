#various functions taken from https://github.com/daveredrum/Pointnet2.ScanNet/blob/master/visualize.py
#other functions added
 
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from .architectures import deepSDFCodedShape
from.utils import funcTimer

from pathlib import Path
import numpy
from PIL import Image
import mcubes
import matplotlib.pyplot as plt
import time
from skimage import measure
import openmesh as om

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
    model_path = os.path.join(dir_model, "floor4square.pth")

    model = deepSDFCodedShape()#.cuda()
    model.load_state_dict(torch.load(model_path, map_location=device))
    # res = 50
    latent = torch.tensor( [-2, 1]).to(device)

    # ptsSample = np.float_([[x, y] 
    #                   for y in  np.linspace(-50, 50, res) 
    #                   for x in np.linspace(-50, 50, res)])

    # pts = torch.Tensor(ptsSample).to(device)


    outputs = model.forward(latent, pts)
    # im = latent_to_image(model, latent)

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

        pixels = model.forward(latent, pts)

        return pixels.view(res, res)


    slices = []


    # endLayer = torch.ones(res, res)
    #add closing slice to first layer 
    # slices.append(endLayer)

    for vector in sliceVectors:
        pixels = createSlice(pts, torch.tensor(vector))
        slices.append(pixels.detach().cpu())

    #add closing slice to last layer
    # slices.append(endLayer)
 
    stacked = np.stack(slices)

    #add alternative meshing
    # contours = []
    # floor_height = 5
    # samples = 30
    # taper = 0

    # for idx, s in enumerate(slices):
        
    #     level = -idx * taper/len(slices)

    #     #after first point and checking previous layer not empty
    #     if(idx > 0 and len(contours[idx - 1]) > 0):
    #         start_point = contours[idx - 1][0][0][:2] #previous layer first coordinate
    #         # slice_contours = extractContours(s, samples, level)
    #         slice_contours = extractContours(s, samples, level, start_point)

    #     else:
    #         slice_contours = extractContours(s, samples, level)

    #     a = []
    #     for contour in slice_contours:
    #         a.append(np.c_[contour, floor_height * idx * np.ones((contour.shape[0], 1))] )# add extra column for height

    #     contours.append(a)


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

    timestr = time.strftime("%d%m%y-%H%M%S")

    fn = "model-" + timestr + ".obj"
    # om.write_mesh(os.path.join(dir_output, fn), mesh)


    # #get the 0 iso surface
    verts, faces = mcubes.marching_cubes(stacked, 0)

    # #normalize verts to 50
    verts[:,1:] /= (res / 50)

    # # Export the result
    timestr = time.strftime("%d%m%y-%H%M%S")


    # fn = "model-" + timestr + ".obj"
    mcubes.export_obj(verts, faces,  os.path.join(dir_output, fn))

    return fn


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

  contour_samples = num_samples//len(contours)

  for contour in contours:

    if(type(start_point) is np.ndarray):
      startIndex = getIndex(contour, start_point)

    if(contour_samples > len(contour)):
        stepSize = 1
    else:
        stepSize = len(contour) // contour_samples

    contour = np.vstack((contour[startIndex:], contour[:startIndex])) #reorganise list to start at nearest pt
    sampled_contour = contour[::stepSize]
    sampled_contour = sampled_contour[:contour_samples ] #bit hacky
    sampled_contour = np.round(sampled_contour, 0)
    sampled_contours.append(sampled_contour)

  return sampled_contours


