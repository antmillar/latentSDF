import time
import numpy as np

from .architectures import deepSDFCodedShape
import torch 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from PIL import Image


def funcTimer(func):

    def timedFunc(*args):

        t0 = time.perf_counter()
        result = func(*args)
        elapsed = time.perf_counter() - t0
        print(f"{func.__name__} : Time Taken - {round(elapsed, 2)} secs")

        return result

    return timedFunc

#pytorch

def load_torch_model(model_path):

    '''Loads the torch model, removes the latents field as not needed for inference'''

    model = deepSDFCodedShape().to(device)

    updated_params = torch.load(model_path, map_location=device)
    updated_params.pop('latents', None)
    new_params = model.state_dict()
    new_params.update(updated_params)
    model.load_state_dict(new_params)

    return model


#constraints

def get_area_covered(sdf):

    '''Determines proportion of sdf inside / total'''

    temp = sdf.cpu().detach().numpy()
    inside = temp < 0
    insideCount = np.sum(inside)
    ptCount = sdf.shape[0]

    coverage = (insideCount / ptCount).item()
    return coverage




def get_site_excess(sdf, site_name, res):

    '''
    Determines whether the sdf stays within the specified site footprint mask
    '''

    temp = sdf.cpu().detach().numpy().reshape(res, res)

    if(site_name == "Canary Wharf"):
        site_footprint = np.array(Image.open('static\img\site_footprint_cw.png').resize((res, res)).convert('L')) / 255.0

    elif(site_name == "St Mary Axe"):
    
        site_footprint = np.array(Image.open('static\img\site_footprint_sm.png').resize((res, res)).convert('L')) / 255.0

    outside_site = site_footprint > 0.1
    within_bldg = temp < 0.1
    out_of_bounds = np.logical_and(within_bldg, outside_site)

    excess = np.sum(out_of_bounds) / (res*res) * 100.0

    return excess

#geometry

def create_rotation_matrix(degrees):
  '''
  Generates a rotation matrix for given angle
  '''

  theta = np.radians(degrees)
  cos, sin = np.cos(theta), np.sin(theta)
  rotation_matrix = torch.tensor(np.array(((cos, -sin), (sin, cos)))).float() #clockwise
  
  return rotation_matrix


#formatting code from https://towardsdatascience.com/beautiful-custom-colormaps-with-matplotlib-5bab3d1f0e72

# def hex_to_rgb(value):
#     '''
#     Converts hex to rgb colours
#     value: string of 6 characters representing a hex colour.
#     Returns: list length 3 of RGB values'''
#     value = value.strip("#") # removes hash symbol if present
#     lv = len(value)
#     return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


# def rgb_to_dec(value):
#     '''
#     Converts rgb to decimal colours (i.e. divides each value by 256)
#     value: list (length 3) of RGB values
#     Returns: list (length 3) of decimal values'''
#     return [v/256 for v in value]
    
# def get_continuous_cmap(hex_list, float_list=None):
#     ''' creates and returns a color map that can be used in heat map figures.
#         If float_list is not provided, colour map graduates linearly between each color in hex_list.
#         If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
#         Parameters
#         ----------
#         hex_list: list of hex code strings
#         float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
#         Returns
#         ----------
#         colour map'''
#     rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
#     if float_list:
#         pass
#     else:
#         float_list = list(np.linspace(0,1,len(rgb_list)))
        
#     cdict = dict()
#     for num, col in enumerate(['red', 'green', 'blue']):
#         col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
#         cdict[col] = col_list
#     cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
#     return cmp

# import matplotlib.colors as mcolors

# divnorm = mcolors.TwoSlopeNorm(vmin=-1,vcenter=0, vmax=1)

# hex_list = ['#ffffff', '#000000', '#ffffff']