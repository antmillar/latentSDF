from scipy import stats
import os
import open3d as o3d
import numpy as np


#These were generated using the get_scannet_stats() function at the bottom

SCANNET_MEANS = [3.08072124, 2.93721497, 0.87943835]
SCANNET_MINS = [3.66360635e-01, 3.56836237e-01, 2.92708193e-03]
SCANNET_MAXS = [5.89, 5.48, 2.41]
SCANNET_VARS = [2.32206428, 2.07387446, 0.40561404]
SCANNET_VOLUME = 75 #average volume in m3
SCANNET_PTNUM = 150000 #average pts per scene
SCANNET_DENSITY = 2000 #average pts per m3

max_density = 500
max_ptCount = 200_000
max_ptCountOnly = 500_000

def normalize_point_cloud(pcd):


    pcd_pts = np.asarray(pcd.points)

    print("info : " + str(pcd))
    print("mean : " + str(stats.describe(pcd_pts).mean))
    print("minmax : " + str(stats.describe(pcd_pts).minmax))
    print("variance : " + str(stats.describe(pcd_pts).variance))

    xMean = stats.describe(pcd_pts).mean[0]
    yMean = stats.describe(pcd_pts).mean[1]
    zMin = stats.describe(pcd_pts).minmax[0][2]

    normalisedPts = np.zeros(pcd_pts.shape)

    #place the model on the plane and approximately center, not really "normalising" here, more just translating 
    normalisedPts[:,0] = ((pcd_pts[:,0] - xMean))
    normalisedPts[:,1] = ((pcd_pts[:,1] - yMean))
    normalisedPts[:,2] = ((pcd_pts[:,2] - zMin))

    print(stats.describe(normalisedPts))

    newPLY = o3d.geometry.PointCloud(pcd)
    newPLY.points = o3d.utility.Vector3dVector(normalisedPts)

    return newPLY

#recursively subsample the point cloud until below threshold
def subsample(pcd, voxel_size):

    next_voxel_size = voxel_size + 0.005
    volume = pcd.get_axis_aligned_bounding_box().volume()
    ptCount = len(pcd.points)
    density = ptCount / volume

    print(f"current point count {ptCount}, density {density}" )

    if ((density > max_density and ptCount > max_ptCount) or ptCount > max_ptCountOnly):
        print(f"too large, so subsampling with voxel size {next_voxel_size}")
        pcd = pcd.voxel_down_sample(next_voxel_size)
        return subsample(pcd, next_voxel_size)
 
    else:
        return pcd


def down_sample(root : str, fn : str, glob : str):

    #load cloud
    pcd = o3d.io.read_point_cloud(root + "/" + fn)

    #get stats
    mean, minmax, variance, volume, ptCount, density = get_stats(root, fn)

    print("beginning subsampling...")

    print(ptCount, density)

    #if the cloud is too density or large subsample it down
    voxel_start = 0.0
    pcd = subsample(pcd, voxel_start)


    return pcd

#remove outliers and normalize
def standardise(pcd, fn, dest):

    #removing outliers
    nnDist = np.mean(pcd.compute_nearest_neighbor_distance())
    print(f"unfiltered point count : {len(pcd.points)}")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=8, std_ratio=2.5)
    print(f"after statistical outliers removed : {len(pcd.points)}")
    pcd, _ = pcd.remove_radius_outlier(nb_points=2, radius=nnDist * 2.5)
    print(f"after radius outliers removed : {len(pcd.points)}")

    #normalizing
    print(f"normalizing file")
    pcd = normalize_point_cloud(pcd)

    #save filtered pcd

    o3d.io.write_point_cloud(dest + "/" + fn[:-4] + "_clean" + ".ply", pcd)
    # getstats(filtered)

#calculate the geometric statistics for a pointcloud
def get_stats(root : str, fn : str):
    
    fileName = os.path.join(root, fn)
    assert(os.path.isfile(fileName))

    pcd = o3d.io.read_point_cloud(fileName)

    pts =  pcd.points
    mean = stats.describe(np.asarray(pts)).mean
    minmax = stats.describe(np.asarray(pts)).minmax
    variance = stats.describe(np.asarray(pts)).variance
    volume = pcd.get_axis_aligned_bounding_box().volume()
    ptCount = len(pts)
    density = ptCount / volume

    return mean, minmax, variance, volume, ptCount, density



#this is a function run over all scans, the directory needs to be updated for local use
#loop over all the scan directories and calculates the statistics for the SCANNET dataset
def get_scannet_stats():

    directory = '/home/anthony/repos/Datasets/ScanNet/scans'
    subfolderList = [item[2] for item in os.walk(directory)]
    files = [file for subFiles in subfolderList for file in subFiles]
    mean = np.zeros([1,3])
    minmax = np.zeros([2,3])
    variance = np.zeros([1,3])

    count = 0

    for file in files:
        if(file[-3:] == "ply"):
            scene_name = file[:12]
            data_folder = os.path.join(directory, scene_name)
            file = os.path.join(data_folder, file)

            pts =  o3d.io.read_point_cloud(file).points
            mean += stats.describe(np.asarray(pts)).mean
            minmax += stats.describe(np.asarray(pts)).minmax
            variance += stats.describe(np.asarray(pts)).variance
            
            count += 1

    print("mean : " + str(mean/count))
    print("minmax : " + str(minmax/count))
    print("variance : " + str(variance/count))


