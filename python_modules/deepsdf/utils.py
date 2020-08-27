import time
import numpy as np
from .architectures import deepSDFCodedShape
import torch 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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