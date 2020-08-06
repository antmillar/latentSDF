import numpy as np
import torch
import matplotlib.pyplot as plt
from.utils import funcTimer
import os
from .architectures import deepSDFCodedShape

cwd = os.getcwd()

dir_image = cwd + '/static/img'

dir_model = cwd + '/torchModels'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

res = 50

ptsSample = np.float_([[x, y] 
                for y in  np.linspace(-50, 50, res) 
                for x in np.linspace(-50, 50, res)])
pts = torch.Tensor(ptsSample).to(device)


def updateLatent(latentBounds):
    """
    Updates the Latent Space Image and Related Data

    Args:
        latentList: list of corners of latent grid [x1, x2, y1, y2]

    """
    #TODO need to check whether corners are valid!
    print("validating bounds...")
    check_bounds(latentBounds)

    #should have model globaL?
    print("loading model...")

    model_path = os.path.join(dir_model, "floor4square.pth")
    model = deepSDFCodedShape().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    interpolate_grid(model, latentBounds)

def check_bounds(latentBounds):
    
    '''
    Ensures bound min < max
    '''

    if(not latentBounds.xMin < latentBounds.xMax or not latentBounds.yMin < latentBounds.yMax):
        raise ValueError('Error: invalid latent space bounds, please ensure Max > Min')

    print("bounds valid")
    
@funcTimer
def interpolate_grid(model, latentBounds, num = 10):
    """Generates an image of the latent space containing seed and interpolated designs

    Args:
        model: pytorch deepsdf model
        corners: the diagonal corners of the latent space
        num: length of side of the grid
    """

    print("generating latent grid...")
    fig, axs = plt.subplots(num, num, figsize=(16, 16))
    fig.tight_layout()

    #TODO need to make this dynamic
    seedLatents = [[0.0,0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]

    #axes
    xAx = np.linspace(latentBounds.xMin, latentBounds.xMax, num)
    yAx = np.linspace(latentBounds.yMax, latentBounds.yMin, num) #bottom left is lowest y

    #find the closest grid item to seed design
    closestLatents = find_seeds(latentBounds, xAx, yAx, seedLatents)

    for index, i in enumerate(xAx):
        for jindex, j in enumerate(yAx) :
        
            latent = torch.tensor( [float(i),  float(j)]).to(device)
            # plot_sdf_from_latent( latent, model.forward, show_axis = False, ax =  axs[index,jindex])

            im = latent_to_image(model, latent)

            if([i, j] in closestLatents):

                im = 255 - im #inverting
                axs[jindex, index].imshow(im, cmap = "winter")

                # #if original show as copper, interpolated show as pink
                if([i, j] in seedLatents):
                    axs[jindex, index].imshow(im, cmap = "copper")

            else:
                # im = Image.fromarray(im * 255.0).convert("RGB") #seems to be a bug with completely white images displaying black, this stops it
                axs[jindex, index].imshow(im, cmap="binary")
        
            axs[jindex, index].axis("off")

        # axs[jindex, index].set_title(np.round(latent.cpu().detach().numpy(), 2), fontsize= 8)

    fig.savefig(os.path.join(dir_image, 'latent_grid.png'))

def find_seeds(latentBounds, xAx, yAx, seedLatents):

    '''
    Finds the nearest point in the latent grid to the seed design, only if within the bounds to a certain tolerance.
    Note: Multiple designs can be nearest to the same grid coordinate

    Args:
        seeds: list of latent vectors of seed designs
        corners: list of the corners of the latent space [x1, x2 ,y1 ,y2]

    Returns: 
        closestLatents: list of the closest latent if within tolerance, else just the seed latent 
    '''

    #tolerance is %5 of the grid shortest side
    tol = 0.05 * min((latentBounds.xMax - latentBounds.xMin), (latentBounds.yMax - latentBounds.yMin) )


    closestLatents = []
    #find the closest coordinates to the latent vector of the seed designs
    for seed in seedLatents:

        #by default set the closest to seed itself
        closest = seed.copy()

        #check if within bounds + tolerance
        inXRange = latentBounds.xMin - tol < seed[0] < latentBounds.xMax + tol
        inYRange = latentBounds.yMin - tol < seed[1] < latentBounds.yMax + tol

        #if so find closest point in latent
        if(inXRange and inYRange):
            closest[0] = min(xAx, key=lambda x:abs(x-seed[0]))
            closest[1] = min(yAx, key=lambda y:abs(y-seed[1]))            

        #if within bounds find the closest
        #if outside but within certain tolerance find closest

        #only approximate the latent if within  the tolerance and within range
        # if(min([ abs(val-seed[0]) for val in xAx ]) < tolerance):

        # if(min([ abs(val-seed[0]) for val in yAx ]) < tolerance):
        #     closest[1] = min(y, key=lambda y:abs(y-seed[1]))

        closestLatents.append(closest)
    return closestLatents

def latent_to_image(model, latent, invert = False):

    sdf = model.forward(latent.to(device), pts)
    pixels = sdf.view(res, res)

    if(invert):
        mask = pixels > 0
    else:
        mask = pixels < 0

    vals = mask.type(torch.uint8) * 255
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
