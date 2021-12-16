import torch



ModelBasePath = ""


''' Apollo models of PointPillars '''

class ApolloModel():
    @staticmethod
    def LoadVoxelEncoder():
        model = torch.jit.load("pts_voxel_encoder.zip")
        print("=" * 10, " LoadVoxelEncoder ", "=" * 10)
        print("model: ", model)
        print("model.code: ", model.code)

    @staticmethod
    def LoadMiddleEncoder():
        model = torch.jit.load("pts_middel_encoder.zip")
        print("=" * 10, " LoadMiddleEncoder ", "=" * 10)
        print("model: ", model)
        print("model.code: ", model.code)    

    @staticmethod
    def LoadBackBone():
        model = torch.jit.load("pts_backbone.zip")
        print("=" * 10, " LoadBackBone ", "=" * 10)
        print("model: ", model)
        print("model.code: ", model.code)    

    @staticmethod
    def LoadNeck():
        model = torch.jit.load("pts_neck.zip")
        print("=" * 10, " LoadNeck ", "=" * 10)
        print("model: ", model)
        print("model.code: ", model.code)    

    @staticmethod
    def LoadBboxHead():
        model = torch.jit.load("pts_bbox_head.zip")
        print("=" * 10, " LoadBboxHead ", "=" * 10)
        print("model: ", model)
        print("model.code: ", model.code)    


''' MMDetection3D models of PointPillars '''
class MMDection3D():
    @staticmethod
    def LoadVoxelEncoder():
        model = torch.jit.load("save_voxel_encoder.zip")
        print("=" * 10, " LoadVoxelEncoder ", "=" * 10)
        print("model: ", model)
        print("model.code: ", model.code)
# def Lodar


if __name__ == "__main__":
    
    ApolloModel.LoadVoxelEncoder()
    MMDection3D.LoadVoxelEncoder()
    # ApolloModel.LoadMiddleEncoder()
    # ApolloModel.LoadBackBone()
    # ApolloModel.LoadNeck()
    # ApolloModel.LoadBboxHead()

