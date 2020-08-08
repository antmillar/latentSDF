import numpy as np
import torch
import matplotlib.pyplot as plt
from.utils import funcTimer
import os
from .architectures import deepSDFCodedShape
from.utils import funcTimer, get_area_covered
import seaborn as sns

cwd = os.getcwd()

dir_image = cwd + '/static/img'

dir_model = cwd + '/torchModels'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

res = 50

ptsSample = np.float_([[x, y] 
                for x in  np.linspace(-50, 50, res) 
                for y in np.linspace(-50, 50, res)])
pts = torch.Tensor(ptsSample).to(device)


class LatentGrid():
    '''
    Stores information about each SDF in the latent grid
    '''
    def __init__(self, dims):

        self.grid = np.empty([dims, dims])
        self.dims = dims

    def addCoverage(self, x, y, coverage):

        self.grid[x, y] = coverage

    def getCoverage(self, x, y):

        return self.grid[x, y]
    
    def reset(self):

        self.__init__(self.dims)

    

grid = LatentGrid(10)

def updateLatent(latentBounds, coverageThreshold = False):
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

    model_path = os.path.join(dir_model, "8floorplans.pth")
    model_path = os.path.join(dir_model, "8floorplans_selflearn.pth")

    model = deepSDFCodedShape().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    interpolate_grid(model, latentBounds, coverageThreshold)

def check_bounds(latentBounds):
    
    '''
    Ensures bound min < max
    '''

    if(not latentBounds.xMin < latentBounds.xMax or not latentBounds.yMin < latentBounds.yMax):
        raise ValueError('Error: invalid latent space bounds, please ensure Max > Min')

    print("bounds valid")
    
@funcTimer
def interpolate_grid(model, latentBounds, coverageThreshold, num = 10):
    """Generates an image of the latent space containing seed and interpolated designs

    Args:
        model: pytorch deepsdf model
        corners: the diagonal corners of the latent space
        num: length of side of the grid
    """
    ##reset the latent grid stored
    grid.reset()
    
    print("generating latent grid...")
    fig, axs = plt.subplots(num, num, figsize=(16, 16))
    # fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    
    '''
    tensor([[-1.6229, -0.2385],
        [-0.5678,  0.5753],
        [-0.6880, -0.6937],
        [ 0.2650, -0.7491],
        [ 1.0502,  0.3029],
        [ 0.3500, -0.3302],
        [ 0.5312,  0.9283],
        [ 0.0736,  0.0258]], device='cuda:0', requires_grad=True)
        '''

    #TODO need to make this dynamic
    seedLatents = [[0.0,0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [0.0, 1.0],  [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    seedLatents = [[-1.6229, -0.2385],
        [-0.5678,  0.5753],
        [-0.6880, -0.6937],
        [ 0.2650, -0.7491],
        [ 1.0502,  0.3029],
        [ 0.3500, -0.3302],
        [ 0.5312,  0.9283],
        [ 0.0736,  0.0258]]
    #axes
    xAx = np.linspace(latentBounds.xMin, latentBounds.xMax, num)
    yAx = np.linspace(latentBounds.yMin, latentBounds.yMax, num)

    #find the closest grid item to seed design
    closestLatents = find_seeds(latentBounds, xAx, yAx, seedLatents)

    for index, i in enumerate(xAx):
        for jindex, j in enumerate(yAx) :
        
            latent = torch.tensor( [float(i),  float(j)]).to(device)
            # plot_sdf_from_latent( latent, model.forward, show_axis = False, ax =  axs[index,jindex])
            
            im, coverage = process_latent(model, latent, grid)

            grid.addCoverage( num - 1 - jindex, index, coverage)

            if([i, j] in closestLatents):
                #starts top left
                axs[num -1 - jindex, index].imshow(im, cmap = "winter")

                # #if original show as copper, interpolated show as pink
                if([i, j] in seedLatents):
                    axs[num - 1 - jindex, index].imshow(im, cmap = "copper")

            else:
                # im = 255 - im #inverting

                #create 4 channel image, set alpha to zero
                im4 = np.stack([im,im,im, np.zeros(im.shape).astype(np.int32)], axis = 2)

                #find black pixels and set their alpha to 1
                blacks = im4[:,:,0] == 0
                im4[:,:,3][blacks] = int(255)
                
                from random import randint
                axs[num -1- jindex, index].imshow(im4, vmin=0, vmax=255, cmap="RdBu")

            axs[num - 1 - jindex, index].axis("off")
            # axs[num - 1 - jindex, index].set_aspect('equal')
    
    values = grid.grid
    if(coverageThreshold):
        threshold = coverageThreshold / 100.0
    else:
        threshold = 0.5
        values = np.ones(grid.grid.shape) * 0.5
    
    fig2, ax2 = plt.subplots(figsize=(16,16))

    #red green diverging palette
    cmap = sns.diverging_palette(5, 150, n=10, as_cmap=True)
    ax2 = sns.heatmap(values, cmap = cmap, cbar=False, center = threshold, vmin = 0.0, vmax = 1.0)
    ax2.axis('off')

    fig2.savefig(os.path.join(dir_image, 'coverage_heatmap.png'), pad_inches=0, bbox_inches = 'tight')
            
    fig.savefig(os.path.join(dir_image, 'latent_grid.png'), transparent=True)

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

        closestLatents.append(closest)
    return closestLatents

def process_latent(model, latent, invert = False):


    #generate sdf from latent vector using model
    sdf = model.forward(latent.to(device), pts)
    # sdf = model.forward(latent, pts)

    coverage = get_area_covered(sdf)
    pixels = sdf.view(res, res)

    if(invert):
        mask = pixels > 0
    else:
        mask = pixels < 0

    vals = mask.type(torch.uint8) * 255
    vals = vals.cpu().detach().numpy()

    # im = Image.fromarray(vals * 255.0).convert("RGB")
    # im.save(os.path.join(dir_output, "test.png"))

    return vals, coverage



def grid_extremes(latentList):
    """Calculates the corners that create a latent space that covers all latent vectors

    Args:
        latentList: list of latent vectors

    Returns:
        corners: two corners of the grid that covers all latents
    """
