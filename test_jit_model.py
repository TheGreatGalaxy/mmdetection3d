from tabnanny import verbose
from turtle import forward
import torch
from torchsummary import summary
import numpy as np


class Torch2Onnx(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        device = torch.device("cuda")
        path = "models/pts_voxel_encoder.zip"
        self.voxel_encoder = torch.jit.load(path, map_location=device)
        print("voxel_encoder: ", self.voxel_encoder)

        path = "models/pts_middle_encoder.zip"
        self.middle_encoder = torch.jit.load(path, map_location=device)
        print("middle_encoder: ", self.middle_encoder)

        path = "models/pts_backbone.zip"
        self.backbone = torch.jit.load(path, map_location=device)
        print("backbone: ", self.backbone)

        path = "models/pts_neck.zip"
        self.neck = torch.jit.load(path, map_location=device)
        print("neck: ", self.neck)

        path = "models/pts_bbox_head.zip"
        self.head = torch.jit.load(path, map_location=device)
        print("head: ", self.head)

    def forward(self, feats, num_points, coors):

        res1 = self.voxel_encoder(feats, num_points, coors)
        # print("res1: ", res1)
        # print("res1: ", res1.shape)

        res2 = self.middle_encoder(res1, coors, torch.tensor([1]))
        # print("res2: ", res2)

        res3 = self.backbone(res2)
        # print("res3: ", res3)

        res4 = self.neck(res3)
        # print("res4: ", res4)

        res5 = self.head(res4)

        return res5


def RunModel():
    device = torch.device("cuda")

    path = "models/pts_voxel_encoder.zip"
    voxel_encoder = torch.jit.load(path, map_location=device)
    print("voxel_encoder: ", voxel_encoder)

    path = "models/pts_middle_encoder.zip"
    middle_encoder = torch.jit.load(path, map_location=device)
    print("middle_encoder: ", middle_encoder)

    path = "models/pts_backbone.zip"
    backbone = torch.jit.load(path, map_location=device)
    print("backbone: ", backbone)

    path = "models/pts_neck.zip"
    neck = torch.jit.load(path, map_location=device)
    print("neck: ", neck)

    path = "models/pts_bbox_head.zip"
    head = torch.jit.load(path, map_location=device)
    print("head: ", head)

    feats = torch.rand(32000, 20, 5).cuda()
    num_points = torch.rand(32000).cuda()
    coors = torch.rand(32000, 4).cuda()
    res1 = voxel_encoder(feats, num_points, coors)
    print("res1: ", res1)
    print("res1: ", res1.shape)

    res2 = middle_encoder(res1, coors, torch.tensor([1]))
    # print("res2: ", res2)

    res3 = backbone(res2)
    # print("res3: ", res3)

    res4 = neck(res3)
    # print("res4: ", res4)

    res5 = head(res4)
    # print("res5: ", res5)
    for ele in res5:
        print(ele.shape)


def ReadPkl():
    # path = "/mmdetection3d/models/pts_voxel_encoder/constants.pkl"
    path = "/mmdetection3d/models/train_6_apollo/mm_voxel_encoder/constants.pkl"
    # fs = open(path, 'rb')
    # data = torch.load(fs, map_location='cpu')
    import numpy as np
    data = np.load(path, mmap_mode='r+', allow_pickle=True)


if __name__ == "__main__":
    # RunModel()
    # ReadPkl()

    feats = torch.rand(32000, 20, 5).cuda()
    num_points = torch.rand(32000).cuda()
    coors = torch.rand(32000, 4).cuda()
    model = Torch2Onnx()

    # input_name = ["feats", "num_points", "coors"]
    # output_name = ["cls", "bbox", "dir"]
    # torch.onnx.export(model, (feats, num_points, coors),
    #                   "pfe_rpn.onnx", verbose=True, input_names=input_name, output_names=output_name, opset_version=11)

    input_name = ["feats", "num_points", "coors"]
    output_name = ["ppfeats"]
    torch.onnx.export(model.voxel_encoder, (feats, num_points, coors),
                      "pfe_rpn.onnx", verbose=True, input_names=input_name, output_names=output_name, opset_version=11)
    # res = model(feats, num_points, coors)
    # for e in res:
    #     print(e.shape)
