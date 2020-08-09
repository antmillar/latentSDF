#various functions taken from https://github.com/daveredrum/Pointnet2.ScanNet/blob/master/visualize.py
#other functions added
 
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from .architectures import deepSDFCodedShape
from.utils import funcTimer, get_area_covered

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
from collections import namedtuple
import random

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
def evaluate(sliceVectors, height, taper):
 
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

    return generateModel(sliceVectors, model, height, taper)



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
def generateModel(sliceVectors, model, numSlices, taper):

    print("generating 3d model...")


    def createSlice(pts, latent):

        sdf = model.forward(latent, pts)

        #need to generate a 50 x 50 sdf for the core

        circle = Circle(np.array([25, 25]), 20)
        coverage = get_area_covered(sdf)

        # print(np.unique(circle.field))

        # intersection = np.maximum(circle.field * -1, sdf.detach().numpy())
        # print(type(intersection))
        # print(intersection.shape)
        return sdf.view(res, res), coverage
        # return torch.tensor(intersection).view(res, res)

    slices = []
    cores = []


    endLayer = torch.ones(res, res)
    #add closing slice to first layer 
    slices.append(endLayer)
    # add = 0.0

    #clamp value
    taper = min(1.0, max(taper, 0.0))
    sliceCount = len(sliceVectors)
    slices_to_taper = int(sliceCount* taper)

    coverages = []

    for vector in sliceVectors[:sliceCount - slices_to_taper]:
        sdf, coverage = createSlice(pts, torch.tensor(vector))
        coverages.append(coverage)

        slices.append(sdf.detach().cpu())

    #need to adjust 2d slices so the distances are accurate in 3d too

    gridDistanceWidth = 4.0 #because the max inside and outside is 1 and -1, 4 = 1 + 1 - - 1 - - 1
    cellSize = gridDistanceWidth / res
    #maybe ignore first layer
    for myIndex, mySlice in enumerate(slices):
      for otherIndex, otherSlice in enumerate(slices):

        #rescaling because performing sqrt later on small numbers otherwise .. underflow
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


    #tapering
    for i in range(slices_to_taper):
      ##cubic aymptote
        shrinkage = 1.0 / (slices_to_taper *slices_to_taper * slices_to_taper)
        slices.append(slices[-slices_to_taper + i].detach().cpu() + shrinkage * i*i*i)



    #add closing slice to last layer
    slices.append(endLayer)
    # cores.append(endLayer)
 
    minCoverage = min(coverages)
    maxCoverage = max(coverages)
    ground_floor = slices[1]
    ground_floor_min = torch.min(ground_floor)
    ground_floor_min_coords = torch.nonzero(ground_floor == torch.min(ground_floor))
    
    stacked = np.stack(slices)

    #generate the isocontours
    contours = []
    floors = []
    floor_labels = []
    floor_height = 2
    samples = 100
    contour_every = 3
    floor_every = 1
    level = 0.0

    ##funny indexing due to extra ends added to close form
    for idx, s in enumerate(slices[1:-1:floor_every]):
        
        # level = -idx * taper/len(slices)

        #after first point and checking previous layer not empty
        if(idx > 0 and len(floors[idx - 1]) > 0):
            start_point = floors[idx - 1][0][0][:2] #previous layer first coordinate
            # slice_contours = extractContours(s, samples, level)
            floor_contours, labels = extractContours(s, samples, level, start_point)

        else:
            floor_contours, labels = extractContours(s, samples, level)

        a = []
        for floor in floor_contours:


            test = np.c_[floor, idx * floor_every * np.ones((floor.shape[0], 1))]
            test[:,2] += 1 #subtract one for the extra base plane
            test[:,2]*= floor_height #need to scale up the height
            test[:,0] -= 25.0
            test[:,1] -= 25.0
            test = test.tolist()
    
            a.append(test)# add extra column for height

        floors.append(a)
        floor_labels.append(labels)


    print(labels)

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
              test[:,2]*= floor_height #need to scale up the height
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
        print(Counter(List))
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


    # core_center = find_core_center(vals)

    minCore = 5
    core_diameter = minCore
    # print("core_center : ", core_center)

    # square = Box(core_diameter, core_diameter, core_center)

    # cores  = [square.field.reshape(res, res) for i in range(numSlices )]



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

      cores  = [square.field.reshape(res, res) for i in range(numSlices)]


      core_centers.append(cores)

    
    intersection = np.minimum.reduce([core for core in core_centers])
    cores = intersection


    # mask2 = vals < 0
    # mask3 = vals < -0.2


    # from PIL import Image
    # im = Image.fromarray(mask2.astype(np.int32) * 255).convert('L')
    # im2 = Image.fromarray(mask3.astype(np.int32) * 255).convert('L')

    # im.save("your_file2.png")
    # im2.save("your_file3.png")




    # circle = Circle(np.array([res//2, res//2]), core_diameter)
    # circle = Circle(core_center, core_diameter)


    # square = Box(core_diameter, core_diameter, core_center)

    # cores  = [square.field.reshape(res, res) for i in range(numSlices)]

      # [52 * circle.field.reshape(res, res)]
    # core = np.stack(core)

    # intersection = np.maximum(core * -1, stacked)
    # stacked = intersection
    #find the level with the lowest how many

    #use this level as core level



    #single core, biggest contour
    # for sdf, lbl in zip(slices[1:-1], floor_labels):



    # #create cores
    # for sdf, lbl in zip(slices[1:-1], floor_labels):

    #     core = sdf.detach().cpu().clone()
    #     num_labels = len(np.unique(lbl))
    #     # print(num_labels)
    #     minDistance = torch.min(sdf)
    #     maxDistance = torch.max(sdf)
    #     targetMin = -0.1

#         for i in range(1, num_labels):
#         #0th group is outside, so don't change for now


#           group = lbl == i
#           # inv = np.invert(group)
          
#           group_min = np.min(np.multiply(group, core).cpu().detach().numpy())  
#           # minDistance = torch.min(sdf)
#           minDistance = group_min
#           maxDistance = torch.max(sdf)
#           # print(i, minDistance, maxDistance)
# #  abs(minDistance - targetMin)
#           max_bump = max(0, (targetMin - group_min))
#           print(max_bump)
#           # max_bump = i * 0.08
#           # print(max_bump - 0.15)
#           # temp = core + 0.15
#           max_bump = np.round(max_bump, 1)
#           max_bump = min(0.20, max_bump)
#           # group_bumped = np.multiply(group, temp)  + np.multiply(inv, core)
#           core += group * max_bump
#           # core = group_bumped
#           # coreScale = 0.5
#           # core = sdf + coreScale * abs(minDistance)
#           # core[core < -0.2] = -0.2
#           # core = sdf + abs(minDistance - targetMin)
#           # print(torch.min(core))
        
#         # group = lbl == 0
        
#         # core += group * max_bump
#         cores.append(core)

        
    # newIdx = len(contours)
    # xSlices = []
    # for i in range(stacked.shape[2]):
    #   xSlices.append(stacked[:,:,i])


    # for idx, s in enumerate(xSlices[2::contour_every*2]):
        
    #     level = -idx * taper/len(xSlices)

    #     #after first point and checking previous layer not empty
    #     if(idx > 0 and len(contours[newIdx + idx - 1]) > 0):
    #         start_point = contours[newIdx + idx - 1][0][0][:2] #previous layer first coordinate
    #         # slice_contours = extractContours(s, samples, level)
    #         slice_contours = extractContours(s, samples, level, start_point)

    #     else:
    #         slice_contours = extractContours(s, samples, level)

    #     a = []
    #     for contour in slice_contours:

    #         #swap this column as we are slicing vertically now
              
    #           test = np.c_[contour, 1 * idx * contour_every*2 * np.ones((contour.shape[0], 1))]


    #           test[:,[0, 2]] = test[:,[2, 0]]
    #           test[:,[0, 1]] = test[:,[1, 0]]
    #           test[:,2]*= floor_height #need to scale up the height
    #           test = test.tolist()
    #           a.append(test)

    #     contours.append(a)



    #filter out empty contours/floors
    floors = [item for item in floors if item != []]

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

    stackedCores = np.stack(cores)
    vertsCore, facesCore = mcubes.marching_cubes(stackedCores, 0)

    # #normalize verts to 50
    verts[:,1:] /= (res / 50)
    vertsCore[:,1:] /= (res / 50)

    # print(verts.shape)
    verts[:,0] -= 1
    verts[:,0] *= floor_height
    verts[:,1] -= 25.0 #translate to threejs space
    verts[:,2] -= 25.0

    # vertsCore[:,0] -= 1
    vertsCore[:,0] *= floor_height
    vertsCore[:,1] -= 25.0 #translate to threejs space
    vertsCore[:,2] -= 25.0


    new_arr = np.ones((faces.shape[0], faces.shape[1] + 1), dtype=np.int32) * 3
    new_arr[:,1:] = faces

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
    core_fn = "core_" + fn
    mcubes.export_obj(verts, faces,  os.path.join(dir_output, fn))
    mcubes.export_obj(vertsCore, facesCore,  os.path.join(dir_output, core_fn))

    # pv.save_meshio(os.path.join(dir_output, fn), surf)

    Details = namedtuple("Details", ["floors", "taper", "maxCoverage", "minCoverage"])
    model_details = Details(numSlices, taper, maxCoverage, minCoverage)

    return fn, contours, floors, model_details


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
  s_binary = s < 0
  labels = measure.label(s_binary)
  startIndex = 0 


  # contour_list = [contour for contour in contours]

  # contours = np.vstack(contour_list)
    
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
                    for x in  np.linspace(start, end, steps) 
                    for y in np.linspace(start, end, steps)])

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
