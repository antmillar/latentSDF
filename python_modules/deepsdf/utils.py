import time
import numpy as np

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