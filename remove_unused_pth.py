import os
import sys

tgt_name = ["epoch_" + str(i) + ".pth" for i in range(24)]

if __name__ == "__main__":
    in_path = "/home/revolution/guangtong/mmdetection3d/checkpoints"
    for fp in os.listdir(in_path):
        if not os.path.isdir(os.path.join(in_path, fp)):
            continue
        for sp in os.listdir(os.path.join(in_path, fp)):
            absp = os.path.join(in_path, fp, sp)
            if os.path.isfile(absp) and sp in tgt_name:
                print("absp: ", absp)
                os.remove(absp)
                if os.path.exists(absp):
                    print("delete file failed!")
                else:
                    print("delete file success!")
