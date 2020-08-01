#this module is largely from https://github.com/daveredrum/Pointnet2.ScanNet/blob/master/lib/dataset.py
#however I have made adaptations and renamed variables for clarity

import os
import sys
import time
import torch
import numpy as np

sys.path.append(".")

NUM_CLASSES = 20 

class ScannetDatasetWholeScene():
    def __init__(self, scene_data, density, max_subvol_points= 8192, is_weighting=True):
        self.scene_data = scene_data
        self.density = density #number of pts per m3 in scan

        self.max_subvol_points = max_subvol_points
        self.is_weighting = is_weighting
        self._load_scene_file()

    def _load_scene_file(self):
        self.scene_points_list = []
        self.semantic_labels_list = []
            
        #load the numpy file
        scene_data = self.scene_data
        scene_data[:, 3:6] /= 255. # normalize the rgb values

        self.scene_points_list.append(scene_data[:, :6])
        self.semantic_labels_list.append(scene_data[:, 7])

        if self.is_weighting:
            labelweights = np.zeros(NUM_CLASSES)
            for seg in self.semantic_labels_list:
                tmp,_ = np.histogram(seg,range(NUM_CLASSES + 1))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        else:
            self.labelweights = np.ones(NUM_CLASSES)

    #index is the sub scene index, not point index
    def __getitem__(self, index):
        start = time.time()
        #point_set_ini contains all the points in the pcd
        point_set_ini = self.scene_points_list[index]
        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)

        ##max coord for each dimension
        coordmax = point_set_ini[:, :3].max(axis=0)
        coordmin = point_set_ini[:, :3].min(axis=0)

        #calculate volume of subvolume 
        volume = self.max_subvol_points / self.density
        sideLength = np.ceil(np.sqrt(volume)) 

        #the issue here is I'm assuming uniform density, clearly the walls and floors have a lot more.

        #dims of the subvolume
        xlength = sideLength #1.5
        ylength = sideLength #1.5

        print(f"xLength : {xlength}")
        print(f"yLength : {ylength}")

        #number of subvolumes
        nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/xlength).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/ylength).astype(np.int32)

        point_sets = list()
        semantic_segs = list()
        sample_weights = list()

        print(f"nsubvolume_x : {nsubvolume_x}")
        print(f"nsubvolume_y : {nsubvolume_y}")

        #loop over subvolumes of size xLength * yLength * height in the scene
        for i in range(nsubvolume_x):

            for j in range(nsubvolume_y):

                #range of the current subvolume
                curmin = coordmin+[i*xlength, j*ylength, 0]
                curmax = coordmin+[(i+1)*xlength, (j+1)*ylength, coordmax[2]-coordmin[2]]
                
                #find points within the subvolume 
                mask = np.all((point_set_ini[:, :3]>=curmin)*(point_set_ini[:, :3]<=curmax), axis=1)
                cur_point_set = point_set_ini[mask,:]
                cur_semantic_seg = semantic_seg_ini[mask]

                #if empty subvolume exit the loop
                if len(cur_semantic_seg) == 0:
                    continue

                print(len(cur_semantic_seg))
                #select 8192 random points from the pt indices in the subvolume
                choice = np.random.choice(len(cur_semantic_seg), self.max_subvol_points, replace=True)

                #get these points from current point set
                point_set = cur_point_set[choice,:] # Nx3
                semantic_seg = cur_semantic_seg[choice] # N

                #this selects 8192 entries from mask (for the whole dataset) based on the first len(cur_semantic_seg)
                mask = mask[choice]

                #if the number of masks hit is less than 1% exit loop
                #I guess this is a way to say that if there are less than 1% of these points in the first len(cur semantic seg) then ignore?
                # if sum(mask)/float(len(mask))<0.01:
                #     continue

                sample_weight = self.labelweights[semantic_seg]
                sample_weight *= mask # N

                point_sets.append(np.expand_dims(point_set,0)) # 1xNx3
                semantic_segs.append(np.expand_dims(semantic_seg,0)) # 1xN
                sample_weights.append(np.expand_dims(sample_weight,0)) # 1xN
   
       
        point_sets = np.concatenate(tuple(point_sets),axis=0)
        semantic_segs = np.concatenate(tuple(semantic_segs),axis=0)
        sample_weights = np.concatenate(tuple(sample_weights),axis=0)

        fetch_time = time.time() - start
        return point_sets, semantic_segs, sample_weights, fetch_time

    def __len__(self):
        #this returns the number of sub scenes in the scene, not the number of points
        return len(self.scene_points_list)


#collate a single scene
def collate_random(data):
    '''
    for ScannetDataset: collate_fn=collate_random
    return: 
        coords               # torch.FloatTensor(B, N, 3)
        feats                # torch.FloatTensor(B, N, 3)
        semantic_segs        # torch.FloatTensor(B, N)
        sample_weights       # torch.FloatTensor(B, N)
        fetch_time           # float
    '''

    # load data (zip the subvolumes)
    (
        point_set, 
        semantic_seg, 
        sample_weight,
        fetch_time 
    ) = zip(*data)

    #dataset contains subvolumes, this collates them into a tensor
    point_set = torch.FloatTensor(point_set)
    semantic_seg = torch.LongTensor(semantic_seg)
    sample_weight = torch.FloatTensor(sample_weight)

    # split points to coords and feats
    coords = point_set[:, :, :3]
    feats = point_set[:, :, 3:]

    # pack
    batch = (
        coords,             # (B, N, 3)
        feats,              # (B, N, 3)
        semantic_seg,      # (B, N)
        sample_weight,     # (B, N)
        sum(fetch_time)          # float
    )

    return batch

#can collate multiple scenes/scene slices

def collate_wholescene(data):
    '''
    for ScannetDataset: collate_fn=collate_random
    return: 
        coords               # torch.FloatTensor(B, C, N, 3)
        feats                # torch.FloatTensor(B, C, N, 3)
        semantic_segs        # torch.FloatTensor(B, C, N)
        sample_weights       # torch.FloatTensor(B, C, N)
        fetch_time           # float
    '''

    # load data (zip the subvolumes)
    (
        point_sets, 
        semantic_segs, 
        sample_weights,
        fetch_time 
    ) = zip(*data)

    #dataset contains subvolumes, this collates them into a tensor
    point_sets = torch.FloatTensor(point_sets)
    print(point_sets.shape)
    semantic_segs = torch.LongTensor(semantic_segs)
    sample_weights = torch.FloatTensor(sample_weights)

    # split points to coords and feats
    coords = point_sets[:, :, :, :3]
    feats = point_sets[:, :, :, 3:]

    # pack
    batch = (
        coords,             # (B, N, 3)
        feats,              # (B, N, 3)
        semantic_segs,      # (B, N)
        sample_weights,     # (B, N)
        sum(fetch_time)          # float
    )

    return batch