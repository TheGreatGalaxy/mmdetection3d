import numpy as np
import os


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
                    xyzio = []
                    for i in split:
                        xyzio.append(float(i))
                    source.extend(xyzio)
                # print("source len: ", len(source))
                source_data = np.array(source).reshape(-1, self.in_dim)
                # print("source data size: ", source_data.shape)
                return source_data
        return np.empty(shape=(0, 0))

    def Run(self):
        if not os.path.exists(self.out_folder_path):
            print("The out folder path: ", self.out_folder_path,
                  " is not exist, will create it ...\n")
            os.makedirs(self.out_folder_path)

        for f in os.listdir(self.in_folder_path):
            abs_path = os.path.join(self.in_folder_path, f)
            name_exten = os.path.splitext(f)
            if os.path.isfile(abs_path) and name_exten[-1] == self.in_postfix:
                data = self.ReadPcdToNumpy(abs_path)
                res = self.ToNuscenesBinWithDeltaTime(data)
                out_path = os.path.join(self.out_folder_path,
                                        name_exten[0] + self.out_postfix)
                res.tofile(out_path)
                print("Save bin file to: ", out_path)

    def RunWithMultiSweeps(self, lidar_idx: list = [0, 1],  sweeps: int = 3, dura_s=500.0):
        '''
        The single_pcd file name is: l${lidar_idx}_${count}_${timestamp_ns}.pcd. The main lidar is l1.
        Args:
            lidar_num: using lidar num, default is 1, 2, which means the front-left and front-right lidar.
            sweeps: each lidar sweeps.
        '''
        assert(len(lidar_idx) >= 2)
        if not os.path.exists(self.out_folder_path):
            print("The out folder path: ", self.out_folder_path,
                  " is not exist, will create it ...\n")
            os.makedirs(self.out_folder_path)

        lidars = [dict() for i in lidar_idx]
        for f in os.listdir(self.in_folder_path):
            abs_path = os.path.join(self.in_folder_path, f)
            name_exten = os.path.splitext(f)
            if os.path.isfile(abs_path) and name_exten[-1] == self.in_postfix:
                # ["l0", count, ts_ns]
                split_name = name_exten[0].split("_")
                split_name.append(abs_path)
                # count : ["l1", count, ts_ns, abs_path]
                idx = int(split_name[0][1]) - 1
                if idx in lidar_idx:
                    lidars[idx][int(split_name[1])] = split_name

        for key, val in lidars[0].items():
            if key < sweeps - 1:
                continue
            base_ts_ns = float(val[2]) * 1e-9
            frames = list()
            indices = range(key, key - sweeps, -1)
            for lidar in lidars:
                for i in indices:
                    if i not in lidar.keys():
                        continue
                    delta_time = base_ts_ns - float(lidar[i][2]) * 1e-9
                    if delta_time < 0 or delta_time > dura_s:
                        continue
                    print("frame: ", lidar[i][3],
                          "delta time s to base: {:.3f}".format(delta_time))
                    xyzi = self.ReadPcdToNumpy(lidar[i][3])
                    xyzid = self.ToNuscenesBinWithDeltaTime(xyzi, delta_time)
                    frames.append(xyzid)
            np.set_printoptions(suppress=True, threshold=np.inf)
            print("before: will merge: ", len(frames), " frames")
            res = np.concatenate(frames, axis=0)
            np.random.shuffle(res)
            out_path = os.path.join(
                self.out_folder_path, str(key) + self.out_postfix)
            res.tofile(out_path)
            print("Save bin file to: ", out_path, "\n")
            # print("res: \n", res)
        pass

    def ToNuscenesBinWithDeltaTime(self, in_np: np.ndarray, delta_time_ms=0.0) -> np.ndarray:
        '''
        Change point cloud to nuscenes bin from pcd. The bin file is n * 5\
            (x, y, z, i, delta_time_ms), and data type is numpy.float32.
        '''
        insert_data = np.full(
            shape=(in_np.shape[0], ), fill_value=delta_time_ms, dtype=np.float32)
        res = np.c_[in_np, insert_data].astype(np.float32)
        return res

    def DisplayPointCloud(self, source_data):
        '''
        params:
            source_data: point cloud, should be numpy 2 dimension variable.
        '''
        import open3d
        point_cloud = open3d.geometry.PointCloud()
        point_cloud.points = open3d.utility.Vector3dVector(source_data[:, :3])
        open3d.visualization.draw_geometries([point_cloud])

    def ReadBinFile(self):
        for f in os.listdir(self.out_folder_path):
            abs_f = os.path.join(self.out_folder_path, f)
            if os.path.splitext(f)[-1] != self.out_postfix:
                continue
            pts = np.fromfile(abs_f, dtype=np.float32).reshape(-1, 5)
            # Display pointcloud.
            self.DisplayPointCloud(pts)


if __name__ == "__main__":
    # in_pcd = "demo/yw_data2/lidar"
    # out_bin = "demo/yw_data2/pcd2bin"
    # pcd2bin = Pcd2bin(in_pcd, out_bin)
    # pcd2bin.Run()
    # pcd2bin.ReadBinFile()

    in_pcd = "demo/yw_data3/single_pcd"
    out_bin = "demo/yw_data3/multi_sweeps"
    pcd2bin = Pcd2bin(in_pcd, out_bin)
    pcd2bin.RunWithMultiSweeps(lidar_idx=[0, 1], sweeps=3, dura_s=0.095)
    # pcd2bin.ReadBinFile()
