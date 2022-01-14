import pdb
import numpy as np
import os
import sys
# from open3d import*

kPcdHeader = "# .PCD v0.7 - Point Cloud Data file format"
kPcdHeaderRows = 11
kInvalidData = 'nan'


class Pcd2bin:
    def __init__(self, in_folder_path, out_folder_path, in_dim=4, in_postfix=".pcd", out_postfix=".bin"):
        self.in_folder_path = in_folder_path
        self.out_folder_path = out_folder_path
        self.in_postfix = in_postfix
        self.out_postfix = out_postfix
        self.in_dim = in_dim

    def ReadPcdToNumpy(self, path):
        # ptf = "/mmdetection3d/demo/pcd/1641461287414859264.pcd"
        source = list()
        with open(path) as f:
            lines = f.readlines()
            if lines[0].find(kPcdHeader) != -1:
                for line in lines[kPcdHeaderRows:]:
                    if line.find(kInvalidData) != -1:
                        continue
                    split = line.strip().split()
                    assert(len(split) == self.in_dim)
                    xyzi = []
                    for i in split:
                        xyzi.append(float(i))
                    source.extend(xyzi)
                # print("source len: ", len(source))
                source_data = np.array(source).reshape(-1, self.in_dim)
                # print("source data size: ", source_data.shape)
                return source_data
        return np.empty(shape=(0, 0))

    def ToNuscenesBin(self, in_np):
        # TODO(): Be cautious about empty pcd.
        insert_data = np.zeros(shape=(in_np.shape[0]))
        res = np.c_(in_np, insert_data)
        return res

    def Run(self):
        if os.path.exists(self.out_folder_path):
            os.makedirs(self.out_folder_path)

        for f in os.listdir(self.in_folder_path):
            abs_path = os.path.join(self.in_folder_path, f)
            name_exten = os.path.splitext(f)
            if os.path.isfile(abs_path) and name_exten[-1] == self.in_postfix:
                data = self.ReadPcdToNumpy(abs_path)
                res = self.ToNuscenesBin(data)
                out_path = os.path.join(self.out_folder_path,
                                        name_exten[0] + "/" + self.out_postfix)
                res.tofile(out_path)
                print("Save bin file to: ", out_path)

    # def DisplayPointCloud(self, source_data):
    #     '''
    #     params:
    #         source_data: point cloud, should be numpy 2 dimension variable.
    #     '''
    #     point_cloud = open3d.geometry.PointCloud()
    #     point_cloud.points = open3d.utility.Vector3dVector(source_data[:, :3])
    #     open3d.visualization.draw_geometries([point_cloud])


if __name__ == "__main__":
    in_pcd = "/home/revolution/guangtong/mmdetection3d/demo/pcd"
    out_bin = "/home/revolution/guangtong/mmdetection3d/demo/pcd/pcd2bin"
    pcd2bin = Pcd2bin(in_pcd, out_bin)
    pcd2bin.Run()
