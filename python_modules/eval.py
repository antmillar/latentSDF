#various functions taken from https://github.com/daveredrum/Pointnet2.ScanNet/blob/master/visualize.py
#other functions added
 
from plyfile import PlyData, PlyElement
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from python_modules.pointnet import pointnet2_semseg as pointnet2_semseg
from python_modules.preprocess.dataset import ScannetDatasetWholeScene, collate_wholescene
from pathlib import Path
import numpy

#static
cwd = os.getcwd()
dir_model = cwd + '/torchModels/'
dir_data = cwd + '/static/models'
dir_output = cwd + '/static/models/outputs'
batch_size = 1

labelMap = [
    (152, 223, 138),		# floor
    (174, 199, 232),		# wall
    (31, 119, 180), 		# cabinet
    (255, 187, 120),		# bed
    (188, 189, 34), 		# chair
    (140, 86, 75),  		# sofa
    (255, 152, 150),		# table
    (214, 39, 40),  		# door
    (197, 176, 213),		# window
    (148, 103, 189),		# bookshelf
    (196, 156, 148),		# picture
    (23, 190, 207), 		# counter
    (247, 182, 210),		# desk
    (219, 219, 141),		# curtain
    (255, 127, 14), 		# refrigerator
    (227, 119, 194),		# bathtub
    (158, 218, 229),		# shower curtain
    (44, 160, 44),  		# toilet
    (112, 128, 144),		# sink
    (82, 84, 163),          # otherfurn
]


#read a ply file to vertex np array
def readPLY(fn):

    print(fn)
    assert(os.path.isfile(fn))

    print("reading PLY file...")
    with open(fn, 'rb') as f:
        
        plydata = PlyData.read(f)
        
        properties = [p.name for p in plydata['vertex'].properties]
        num_verts = plydata['vertex'].count
        num_properties = len(properties)

        #add vertex data
        points = np.zeros(shape=[num_verts, num_properties], dtype=np.float32)
        points[:,0] = plydata['vertex'].data['x']
        points[:,1] = plydata['vertex'].data['y']
        points[:,2] = plydata['vertex'].data['z']

        #add RGB data if present, otherwise defaults to zero
        if(num_properties == 6 & all(x in ['red' , 'green', 'blue'] for x in properties)):

            points[:,3] = plydata['vertex'].data['red']
            points[:,4] = plydata['vertex'].data['green']
            points[:,5] = plydata['vertex'].data['blue']
    
    return points


#add two blank columns for input into the model
def add_blank_cols(vertices : np.array):
    data = np.zeros(shape=[vertices.shape[0], 8])
    data[:,:6] = vertices[:,:6]
    return data


#save the numpy data locally
def save(scene_data : np.array):
    np.save(dir_data + "/numpy" , scene_data)

#load a point cloud and pass through the model
def evaluate(input_path : str, fn : str, density : float):

    print("reading input PLY file...")
    print(input_path + "/" +  fn)
    scene_data = readPLY(input_path + "/" +  fn)
    scene_data = add_blank_cols(scene_data)

    print(f"shape : {scene_data.shape}")

    print("preparing data...")
    dataset = ScannetDatasetWholeScene(scene_data, density)
    dataloader = DataLoader(dataset, batch_size= batch_size, collate_fn=collate_wholescene)

    print("loading model...")
    model_path = os.path.join(dir_model, "model.pth")
    model = pointnet2_semseg.get_model(num_classes=20, is_msg = False).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print("labelling points...")
    pred = predict_label(model, dataloader)
    
    print("saving PLY file...")
    save_to_PLY(fn, pred)
    

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
