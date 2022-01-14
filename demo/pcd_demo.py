# Copyright (c) OpenMMLab. All rights reserved.
import os
from posixpath import abspath
import sys
import time
from argparse import ArgumentParser

from mmdet3d.apis import inference_detector, init_model, show_result_meshlab


def main():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.0, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo/temp', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    pcd_lists = list()
    if os.path.isdir(args.pcd):
        for f in os.listdir(args.pcd):
            abs_path = os.path.join(args.pcd, f)
            if os.path.isfile(abs_path):
                pcd_lists.append(abs_path)
        print("====== In directory {}, has {} frames pointclouds. ======".format(args.pcd, len(pcd_lists)))
    elif os.path.isfile(args.pcd):
        pcd_lists.append(args.pcd)

    # count = 0
    for idx, pcd in enumerate(pcd_lists):
        # count += 1
        # if count > 5:
        #     exit()
        # test a single image
        print("====== now is processing frame {}, pcd file: {}. ======".format(idx, pcd))
        start  = time.time()
        result, data = inference_detector(model, pcd)
        infer_time = time.time()
        print("inference time seconds: ", infer_time - start)
        # show the results
        show_result_meshlab(
            data,
            result,
            args.out_dir,
            args.score_thr,
            show=args.show,
            snapshot=args.snapshot,
            task='det')
        render_time = time.time()
        print("render time seconds: ", render_time - infer_time )


if __name__ == '__main__':
    main()
