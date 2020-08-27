import numpy as np
import torch
import matplotlib.pyplot as plt
from.utils import funcTimer
import os
from .architectures import deepSDFCodedShape
from.utils import funcTimer, get_area_covered, get_site_excess
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


cwd = os.getcwd()

dir_image = cwd + '/static/img'
dir_image_designs = dir_image + '/designs/'
dir_model = cwd + '/torchModels'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LatentGrid():
    '''
    Stores information about each SDF in the latent grid
    '''
    def __init__(self, dims):

        self.grid = np.empty([dims, dims])
        self.dims = dims

    def addValue(self, x, y, value):

        self.grid[x, y] = value

    def getValue(self, x, y):

        return self.grid[x, y]
    
    def reset(self):

        self.__init__(self.dims)

    

grid = LatentGrid(10)
site_grid = LatentGrid(10)

def updateLatent(latentBounds,  model_path, latents, site_name = "St Mary Axe",  coverage_threshold = False, check_site_boundary = False):
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
    # model_path = os.path.join(dir_model, "8floorplans.pth")
    # model_path = os.path.join(dir_model, "8floorplans_selflearn.pth")


    model = deepSDFCodedShape().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    interpolate_grid(model, latentBounds, latents, site_name,  coverage_threshold, check_site_boundary)

def check_bounds(latentBounds):
    
    '''
    Ensures bound min < max
    '''

    if(not latentBounds.xMin < latentBounds.xMax or not latentBounds.yMin < latentBounds.yMax):
        raise ValueError('Error: invalid latent space bounds, please ensure Max > Min')

    print("bounds valid")
    
@funcTimer
def interpolate_grid(model, latentBounds, latents : np.array, site_name, coverage_threshold, check_site_boundary, num = 10):
    """Generates an image of the latent space containing seed and interpolated designs

    Args:
        model: pytorch deepsdf model
        corners: the diagonal corners of the latent space
        num: length of side of the grid
    """
    ##reset the latent grid stored
    grid.reset()
    site_grid.reset()
    
    print("generating latent grid...")
    fig, axs = plt.subplots(num, num, figsize=(16, 16))
    # fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    

    #axes
    xAx = np.linspace(latentBounds.xMin, latentBounds.xMax, num)
    yAx = np.linspace(latentBounds.yMin, latentBounds.yMax, num)

    #find the closest grid item to seed design
    closestLatents =  find_seeds(latentBounds, xAx, yAx, latents)

    for index, i in enumerate(xAx):
        for jindex, j in enumerate(yAx) :
        
            latent = torch.tensor( [float(i),  float(j)]).to(device)
            # plot_sdf_from_latent( latent, model.forward, show_axis = False, ax =  axs[index,jindex])
            
            im, coverage, site_excess = process_latent(model, latent, site_name, grid)

            grid.addValue( num - 1 - jindex, index, coverage)
            site_grid.addValue( num - 1 - jindex, index, site_excess)


            if([i, j] in closestLatents):
                #starts top left
                axs[num -1 - jindex, index].imshow(im, cmap = "copper")

                # # #if original show as copper, interpolated show as pink
                # if([i, j] in latents):
                #     axs[num - 1 - jindex, index].imshow(im, cmap = "copper")

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
    

    #####
    grid_is_transparent = False

    if(coverage_threshold != False or check_site_boundary != False):

        create_heatmap(coverage_threshold)
        grid_is_transparent = True
            
    fig.savefig(os.path.join(dir_image, 'latent_grid.png'), transparent=grid_is_transparent)

def create_heatmap(coverage_threshold):

    #by default create heatmap for site extents, where 1% is the threshold, otherwise coverage threshold
    values = site_grid.grid
    threshold = 1
    start_color = 150
    end_color = 5
    vmin = 0.0
    vmax = 10

    if(coverage_threshold != False):
        threshold = coverage_threshold / 100.0
        grid_is_transparent = True
        values = grid.grid
        start_color = 5
        end_color = 150
        vmin = 0.0
        vmax = 1.0


    fig2, ax2 = plt.subplots(figsize=(16,16))

    #red green diverging palette
    cmap = sns.diverging_palette(start_color, end_color, n=10, as_cmap=True)

    ax2 = sns.heatmap(values, cmap = cmap, cbar=False, center = threshold, vmin = vmin, vmax = vmax)
    ax2.axis('off')

    fig2.savefig(os.path.join(dir_image, 'constraint_heatmap.png'), pad_inches=0, bbox_inches = 'tight')
    ######


def find_seeds(latentBounds, xAx, yAx, latents):

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
    for seed in latents:

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

def process_latent(model, latent, site_name , invert = False,  res = 50):

    ptsSample = np.float_([[x, y] 
                for x in np.linspace(-50, 50, res) 
                for y in np.linspace(50, -50, res)])
    pts = torch.Tensor(ptsSample).to(device)

    #generate sdf from latent vector using model
    sdf = model.forward(latent.to(device), pts)
    # sdf = model.forward(latent, pts)

    coverage = get_area_covered(sdf)
    site_excess = get_site_excess(sdf, site_name)
    pixels = sdf.view(res, res)
    # print(site_excess)

    if(invert):
        mask = pixels > 0
    else:
        mask = pixels < 0

    vals = mask.type(torch.uint8) * 255
    vals = vals.cpu().detach().numpy()

    return vals, coverage, site_excess

def prepare_analysis(model_path, latents):
    '''
    Generates images for a set of latent vectors
    '''

    model = deepSDFCodedShape().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    images = [process_latent(model, torch.tensor(latent), res = 100)[0] for  latent in latents]

    for im, latent in zip(images, latents):
        plt.imsave(dir_image_designs + str(latent) + ".png", im, cmap = "binary" )
    
    create_distance_plot(latents)

def create_distance_plot(latents):
    '''
    Generates a plot of the designs and their relative positions
    '''
    def getImage(path):
        return OffsetImage(plt.imread(path), zoom=.3)

    paths = [dir_image_designs + str(latent) + ".png" for latent in latents]

    latent_codes = np.array(latents)
    print(latents)
    x = latent_codes[:,0]
    y = latent_codes[:,1]

    fig, ax = plt.subplots(figsize = (12, 12))
    plt.xlim(min(x) - 0.25,  max(x) + 0.25)
    plt.ylim(min(y) - 0.25,  max(y) + 0.25)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)   

    ax.scatter(x, y) 

    for x0, y0, path in zip(x, y, paths):
        ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
        ax.add_artist(ab)

    fig.savefig(dir_image_designs + "distance.jpg")
    
def calculate_distances(index, annotations):
    """
    Calculates distances relative to a chosen seed design
    """
    print(annotations)

    #index minus one because using 1-indexing in design gui
    target_latent = annotations[index - 1, 1]

    def euclidean_dist(l1, l2):

        return np.round(np.sqrt(pow(l2[0] - l1[0], 2) + pow(l2[1] - l1[1], 2)), 2)

    distances = []
    for latent in annotations[:,1]:
        distances.append(euclidean_dist(latent, target_latent))

    sortedOrder = np.array(distances).argsort()
    
    newArray = annotations.copy()
    newArray = np.c_[ newArray, np.array(distances)]     

    return newArray[sortedOrder]


def grid_extremes(latentList):
    """Calculates the corners that create a latent space that covers all latent vectors

    Args:
        latentList: list of latent vectors

    Returns:
        corners: two corners of the grid that covers all latents
    """
