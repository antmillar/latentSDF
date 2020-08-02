#various functions taken from https://github.com/daveredrum/Pointnet2.ScanNet/blob/master/visualize.py
#other functions added
 
# from plyfile import PlyData, PlyElement
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from .architectures import deepSDFCodedShape
from pathlib import Path
import numpy
from PIL import Image
import mcubes
import matplotlib.pyplot as plt
import time

#static
cwd = os.getcwd()
dir_image = cwd + '/static/img'
dir_model = cwd + '/torchModels'
dir_data = cwd + '/static/models'
dir_output = cwd + '/static/models/outputs'
batch_size = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

size = 100
ptsSample = np.float_([[x, y] 
                for y in  np.linspace(-50, 50, size) 
                for x in np.linspace(-50, 50, size)])
                    
#save the numpy data locally
def save(scene_data : np.array):
    np.save(dir_data + "/numpy" , scene_data)

#load a point cloud and pass through the model
def evaluate(sliceVectors):

    # print("reading input PLY file...")
    # print(input_path + "/" +  fn)
    # scene_data = readPLY(input_path + "/" +  fn)
    # scene_data = add_blank_cols(scene_data)

    # print(f"shape : {scene_data.shape}")

    # print("preparing data...")
    # dataset = ScannetDatasetWholeScene(scene_data, density)
    # dataloader = DataLoader(dataset, batch_size= batch_size, collate_fn=collate_wholescene)

 
    print("loading model...")
    model_path = os.path.join(dir_model, "floors4.pth")

    model = deepSDFCodedShape()#.cuda()
    model.load_state_dict(torch.load(model_path, map_location=device))
    # res = 50
    # latent = torch.tensor( [1, 0]).to(device)

    # ptsSample = np.float_([[x, y] 
    #                   for y in  np.linspace(-50, 50, res) 
    #                   for x in np.linspace(-50, 50, res)])

    # pts = torch.Tensor(ptsSample).to(device)


    # outputs = model.forward(latent, pts)
    # im = latent_to_image(model, latent)

    # print("labelling points...")
    # pred = predict_label(model, dataloader)
    
    # print("saving PLY file...")
    # save_to_PLY(fn, pred)

    # generateModel(sliceVectors, model, numSlices, res)



    print("complete")

def updateLatent(latentBounds):

    corners = np.array([latentBounds[0], latentBounds[1]]), np.array([latentBounds[2], latentBounds[3]])
    #should have model globaL?
    print("loading model...")
    model_path = os.path.join(dir_model, "floors4.pth")

    model = deepSDFCodedShape()#.cuda()
    model.load_state_dict(torch.load(model_path, map_location=device))

    interpolate_grid(model, corners)



def funcTimer(func):

    def timedFunc(*args):

        t0 = time.perf_counter()
        result = func(*args)
        elapsed = time.perf_counter() - t0
        print(f"{func.__name__} : Time Taken - {np.round(elapsed, 2)} secs")

        return result

    return timedFunc

def latent_to_image(model, latent, invert = False):

    coord = torch.Tensor(ptsSample).to(device)
    out = model.forward(latent.to(device), coord)
    pixels = out.view(size, size)

    if(invert):
        mask = pixels < 0
    else:
        mask = pixels > 0
    vals = mask.type(torch.uint8) 
    vals = vals.cpu().detach().numpy()

    # im = Image.fromarray(vals * 255.0).convert("RGB")
    # im.save(os.path.join(dir_output, "test.png"))

    return vals


def grid_extremes(latentList):
    """Calculates the corners that create a latent space that covers all latent vectors

    Args:
        latentList: list of latent vectors

    Returns:
        corners: two corners of the grid that covers all latents
    """
    
@funcTimer
def interpolate_grid(model, corners, num = 10):
    """Generates an image of the latent space containing seed and interpolated designs

    Args:
        model: pytorch deepsdf model
        corners: the diagonal corners of the latent space
        num: length of side of the grid
    """

    print("generating latent grid...")
    fig, axs = plt.subplots(num, num, figsize=(16, 16))
    fig.tight_layout()

    #axes
    x = np.linspace(corners[0][0], corners[0][1], num)
    y = np.linspace(corners[1][1], corners[1][0], num) #bottom left is lowest y

    for index, i in enumerate(x):
        for jindex, j in enumerate(y) :
        
            latent = torch.tensor( [float(i),  float(j)]).to(device)
            # plot_sdf_from_latent( latent, model.forward, show_axis = False, ax =  axs[index,jindex])

            im = latent_to_image(model, latent)

            if([i, j] in [[0.0,0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]):

                axs[jindex, index].imshow(im, cmap = "copper")
            else:

                axs[jindex, index].imshow(im, cmap = "gray")
        
            axs[jindex, index].axis("off")

        # axs[jindex, index].set_title(np.round(latent.cpu().detach().numpy(), 2), fontsize= 8)

    fig.savefig(os.path.join(dir_image, 'latent_grid.png'))


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
def generateModel(sliceVectors, model, numSlices, res):

    print("generating 3d model...")

    latentStart = torch.tensor( [1, 0.5]).to(device)
    latentEnd = torch.tensor( [2, 0]).to(device)

    ptsSample = np.float_([[x, y] 
                        for y in  np.linspace(-50, 50, res) 
                        for x in np.linspace(-50, 50, res)])

    pts = torch.Tensor(ptsSample).to(device)

    def createSlice(pts, latent):

        pixels = model.forward(latent, pts)

        return pixels.view(res, res)


    latentRange = latentEnd - latentStart 
    latentStep = latentRange.div(numSlices)
    slices = []

    for vector in sliceVectors:
        pixels = createSlice(pts, torch.tensor(vector))
        slices.append(pixels.detach().cpu())

    stacked = np.stack(slices)

    #get the 0 iso surface
    verts, faces = mcubes.marching_cubes(stacked, 0)

    #normalize verts to 50
    verts[:,1:] /= (res / 50)

    # Export the result
    mcubes.export_obj(verts, faces,  os.path.join(dir_output, "test.obj"))


