import time
import numpy as np
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

def get_site_excess(sdf, site_name):

    '''
    Tests whether the sdf stays within the specified site footprint mask
    '''

    temp = sdf.cpu().detach().numpy().reshape(50, 50)

    if(site_name == "Canary Wharf"):
        site_footprint = np.array(Image.open('static\img\site_footprint_cw.png').resize((50, 50)).convert('L')) / 255.0

    elif(site_name == "St Mary Axe"):
    
        site_footprint = np.array(Image.open('static\img\site_footprint_sm.png').resize((50, 50)).convert('L')) / 255.0

    outside_site = site_footprint > 0.1
    within_bldg = temp < 0.1
    out_of_bounds = np.logical_and(within_bldg, outside_site)

    excess = np.sum(out_of_bounds) / (50*50) * 100.0

    return excess