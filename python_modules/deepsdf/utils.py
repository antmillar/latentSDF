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


def get_area_covered(sdf):

    #count the proportion of values that are negative
    temp = sdf.cpu().detach().numpy()
    inside = temp < 0
    insideCount = np.sum(inside)
    ptCount = sdf.shape[0]

    coverage = (insideCount / ptCount).item()
    return coverage


def load_torch_model(model_path):

    model = deepSDFCodedShape().to(device)

    #update params to remove the latent vector, not needed for inference
    updated_params = torch.load(model_path, map_location=device)
    updated_params.pop('latents', None)
    new_params = model.state_dict()
    new_params.update(updated_params)
    model.load_state_dict(new_params)

    return model


def get_site_excess(sdf, site_name, res):

    '''
    Tests whether the sdf stays within the specified site footprint mask
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

