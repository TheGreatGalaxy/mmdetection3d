import pdb
import numpy as np
from open3d import*

#  test self pcd.
# ptf = "/mmdetection3d/demo/pcd/1641461287414859264.pcd"
# source = list()
# with open(ptf) as f:
#     for line in f:
#         split = line.strip().split()
#         if len(split) == 4:
#             xyzi = []
#             for i in split:
#                 if i != 'nan':
#                     xyzi.append(float(i))
#             if len(xyzi) == 4:
#                 source.extend(xyzi)
# import numpy as np
# source_data = np.array(source).reshape(-1, 4)
# source_data[:, -1] = 0
# pts = [torch.tensor(source_data, dtype=torch.float,
#                     device=pts[0].device)]


# source_data = np.load('curtain_0088.npy')[:,0:3]  #10000x3
# source_data = np.random.randint(-100, high=100, size=(1000, 3))
ptf = "/mmdetection3d/demo/pcd/1641461287414859264.pcd"
source = list()
with open(ptf) as f:
    for line in f:
        split = line.strip().split()
        if len(split) == 4:
            xyzi = []
            for i in split:
                if i != 'nan':
                    xyzi.append(float(i))
            if len(xyzi) == 4:
                source.extend(xyzi)
print("source len: ", len(source))
pdb.set_trace()
source_data = np.array(source).reshape(-1, 4)[:, :-1]
print("source data size: ", source_data.shape)

point_cloud = open3d.geometry.PointCloud()
point_cloud.points = open3d.utility.Vector3dVector(source_data)
open3d.visualization.draw_geometries([point_cloud])
