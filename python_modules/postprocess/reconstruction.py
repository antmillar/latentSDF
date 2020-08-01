import open3d as o3d
import numpy as np

#mapping according to scannet paper
filterMap = {

"floor"          :   [152., 223., 138.],
"wall"           :   [174., 199., 232.],
"cabinet"        :   [31., 119., 180.],
"bed"            :   [255., 187., 120.],
"chair"          :   [188., 189., 34.],
"sofa"           :   [140., 86., 75.],
"table"          :   [255., 152., 150.],
"door"           :   [214., 39., 40.],
"window"         :   [197., 176., 213.],
"bookshelf"      :   [148., 103., 189.],
"picture"        :   [196., 156., 148.],
"counter"        :   [23., 190., 207.],
"desk"           :   [247., 182., 210.],
"curtain"        :   [219., 219., 141.],
"refrigerator"   :   [255., 127., 14.],
"bathtub"        :   [227., 119., 194.],
"shower curtain" :   [158., 218., 229.],
"toilet"         :   [44., 160., 44.],
"sink"           :   [112., 128., 144.],
"other"          :   [82., 84., 163.],
"black"          :   [0., 0., 0.],
}


def save_to_mesh(folder, labelFolder, dest, fn, filters, reconstruction_method = "ballpivot"):

    print("loading PLY files...")
    
    pcd = o3d.io.read_point_cloud(folder + "/" + fn)
    pcd_labelled = o3d.io.read_point_cloud(labelFolder + "/" + fn[:-4] + "_labels.ply")

    print("estimating normals...")
    pcd.estimate_normals()

    #ball pivot reconstruction - doesn't required require handling MTL files
    if(reconstruction_method == "ballpivot"):

        print("calculating ball pivot reconstruction...")

        nnDist = np.mean(pcd.compute_nearest_neighbor_distance())
        print(f"nearest neighbour distance : {nnDist}")
        dim = 2.5 * nnDist 
        ball_radius = np.array([dim, dim, dim ])
        ball_radius = o3d.utility.DoubleVector(ball_radius)

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, ball_radius)
 

    #use poisson reconstruction - gives a slightly smoother reconstruction, but as it's probabilistic needs texture mapping
    #haven't implemented MTL file handling
    elif(reconstruction_method == "poisson"):

        print("calculating poisson reconstruction...")
        
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth = 8)

        #remove any points with density below threshold
        print("filtering densities...")
        mask_densities = densities < np.quantile(densities, 0.2)
        mesh.remove_vertices_by_mask(mask_densities)


    #paint the colors from the labelled pcd to the mesh reconstructed from the original higher res pcd

    print("painting colors...")

    mesh = paint_colors(mesh, pcd_labelled)

    #if filters given filter the mesh by label
    if(len(filters) > 0):
        print("filtering vertices...")
        mesh = filter_mesh(mesh, filters)


    print("saving mesh...")
    fn_save = dest + "/" + fn[:-4] + ".obj"
    o3d.io.write_triangle_mesh(fn_save, mesh)
    
    print(f"mesh saved at {fn_save}")



def paint_colors(mesh, pcd_labelled, iterations = 3):

    #get points and colors
    pts_labels = np.asarray(pcd_labelled.points)
    pts_labels_colors = np.asarray(pcd_labelled.colors)
    pts_mesh = np.asarray(mesh.vertices)
    pts_mesh_colors = np.asarray(mesh.vertex_colors)

    #round the coordinates as using different meshes so slight differences
    pts_labels_round = pts_labels.round(2)
    pts_mesh_round = pts_mesh.round(2)

    #hash the coords for comparison
    hash_labels = [hash(str(pts_labels_round[i][0]) + str(pts_labels_round[i][1]) + str(pts_labels_round[i][2])) for i in range(pts_labels_round.shape[0])]
    hash_mesh = [hash(str(pts_mesh_round[i][0]) + str(pts_mesh_round[i][1]) + str(pts_mesh_round[i][2])) for i in range(pts_mesh_round.shape[0])]
    np_labels = np.array(hash_labels)
    np_mesh = np.array(hash_mesh)

    #find the coordinates that are shared
    intersect_labels = np.intersect1d(np_labels, np_mesh, return_indices=True)[1]
    intersect_mesh = np.intersect1d(np_labels, np_mesh, return_indices=True)[2]

    #create new array for painted colors
    painted_colors = np.zeros((pts_mesh_colors.shape))
    painted_colors[intersect_mesh] = pts_labels_colors[intersect_labels]

    #get adjacency indices for painting neighbours
    mesh.compute_adjacency_list()
    mesh_adj = mesh.adjacency_list

    #loop over the shared points in the mesh and paint colors to neighbours
    for meshIdx in intersect_mesh:
        for neighbourIdx in list(iter(mesh_adj[meshIdx])):
            painted_colors[neighbourIdx] = painted_colors[meshIdx]

    iterate = 0
    no_color = np.array([0, 0, 0])

    #repeat the painting process for better coverage / note: probably could reverse the logic and copy onto black cells
    while iterate < iterations:

        mask = painted_colors != no_color
        mask = np.all(mask, axis = 1)
        painted_pts = np.argwhere(mask == True).squeeze(1)

        for meshIdx in painted_pts:
            for neighbourIdx in list(iter(mesh_adj[meshIdx])):
                painted_colors[neighbourIdx] = painted_colors[meshIdx]

        iterate += 1


    mesh.vertex_colors = o3d.utility.Vector3dVector( np.asarray(painted_colors) )

    return mesh

#filters out points from mesh based on labels
def filter_mesh(mesh, filters):

    colors = np.asarray(mesh.vertex_colors) * 255.0

    points = np.asarray(mesh.vertices)

    mask = np.zeros(colors.shape[0], dtype=bool)

    filters.append("black")
    #create boolean mask by looping over vertex colors with filters
    for filter in filters:
        mask += np.all(colors == np.array(filterMap[filter]), axis = 1)

    mesh.remove_vertices_by_mask(mask)
   
    return mesh