# Copyright (c) OpenMMLab. All rights reserved.
from binascii import Incomplete
from tabnanny import verbose
from turtle import update
import mmcv
import torch
import warnings
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
import os
from torch.nn import functional as F

from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from mmdet3d.ops import Voxelization, Pillar
from mmdet.core import multi_apply
from mmdet.models import DETECTORS
from .. import builder
from .base import Base3DDetector


@DETECTORS.register_module()
class PointPillarsCore(Base3DDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self,
                 #  pts_voxel_layer=None,
                 #  pts_voxel_encoder=None,
                 #  pts_middle_encoder=None,
                 #  pts_fusion_layer=None,
                 #  img_backbone=None,
                 pts_backbone=None,
                 #  img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 #  img_roi_head=None,
                 #  img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(PointPillarsCore, self).__init__(init_cfg=init_cfg)

        # if pts_voxel_layer:
        #     self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        # if pts_voxel_encoder:
        #     self.pts_voxel_encoder = builder.build_voxel_encoder(
        #         pts_voxel_encoder)
        # if pts_middle_encoder:
        #     self.pts_middle_encoder = builder.build_middle_encoder(
        #         pts_middle_encoder)
        if pts_backbone:
            self.pts_backbone = builder.build_backbone(pts_backbone)
        if pts_neck is not None:
            self.pts_neck = builder.build_neck(pts_neck)
        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if pretrained is None:
            pts_pretrained = None
        elif isinstance(pretrained, dict):
            pts_pretrained = pretrained.get('pts', None)
        else:
            raise ValueError(
                f'pretrained should be a dict, got {type(pretrained)}')
        if self.with_pts_backbone:
            if pts_pretrained is not None:
                warnings.warn('DeprecationWarning: pretrained is a deprecated \
                    key, please consider using init_cfg')
                self.pts_backbone.init_cfg = dict(
                    type='Pretrained', checkpoint=pts_pretrained)

    @property
    def with_pts_bbox(self):
        """bool: Whether the detector has a 3D box head."""
        return hasattr(self,
                       'pts_bbox_head') and self.pts_bbox_head is not None

    @property
    def with_pts_backbone(self):
        """bool: Whether the detector has a 3D backbone."""
        return hasattr(self, 'pts_backbone') and self.pts_backbone is not None

    @property
    def with_pts_neck(self):
        """bool: Whether the detector has a neck in 3D detector branch."""
        return hasattr(self, 'pts_neck') and self.pts_neck is not None

    @property
    def with_voxel_encoder(self):
        """bool: Whether the detector has a voxel encoder."""
        return hasattr(self,
                       'voxel_encoder') and self.voxel_encoder is not None

    @property
    def with_middle_encoder(self):
        """bool: Whether the detector has a middle encoder."""
        return hasattr(self,
                       'middle_encoder') and self.middle_encoder is not None

    def aug_test(self, img, img_metas, **kwargs):
        '''Just override abstract method inherited from parent class'''
        raise NotImplementedError

    def extract_feat(self, img):
        '''Just override abstract method inherited from parent class'''
        raise NotImplementedError

    def forward(self, input_canvas):
        '''
        Args:
            input_canvas: scattered pillars canvas, [N, P, C]
        '''
        return self.extract_pts_feat(input_canvas)

    def extract_pts_feat(self, input_canvas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None

        # print("before voxelize pts shape: \n", pts[0].shape)
        # before voxelize pts shape:  torch.Size([58103, 4])

        # voxels, num_points, coors = self.voxelize(pts)
        # print("after voxelize voxels:", voxels.shape)
        # after voxelize voxels: torch.Size([3151, 20, 4])
        # print("after voxelize num_points: ", num_points.shape)
        # after voxelize num_points:  torch.Size([3151])
        # print("after voxelize coors: ", coors.shape)
        # after voxelize coors:  torch.Size([3151, 4])

        # voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)

        # print("after HardVFE voxel_feature: \n", voxel_features.shape)
        # after HardVFE voxel_feature:  torch.Size([3151, 64])

        # batch_size = coors[-1, 0] + 1
        # x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(input_canvas)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        outs = self.pts_bbox_head(x)
        return outs

    # def forward_test(self, **kwargs):
    #     """
    #     Args:
    #         points (list[torch.Tensor]): the outer list indicates test-time
    #             augmentations and inner torch.Tensor should have a shape NxC,
    #             which contains all points in the batch.
    #         img_metas (list[list[dict]]): the outer list indicates test-time
    #             augs (multiscale, flip, etc.) and the inner list indicates
    #             images in a batch
    #         img (list[torch.Tensor], optional): the outer
    #             list indicates test-time augmentations and inner
    #             torch.Tensor should have a shape NxCxHxW, which contains
    #             all images in the batch. Defaults to None.
    #     """
    #     print("Is this forward test function")
    #     return self.simple_test(**kwargs)

    # def forward_train(self,
    #                   voxels=None, num_points=None, coors=None,
    #                   img_metas=None,
    #                   gt_bboxes_3d=None,
    #                   gt_labels_3d=None,
    #                   gt_labels=None,
    #                   gt_bboxes=None,
    #                   img=None,
    #                   proposals=None,
    #                   gt_bboxes_ignore=None):
    #     """Forward training function.

    #     Args:
    #         points (list[torch.Tensor], optional): Points of each sample.
    #             Defaults to None.
    #         img_metas (list[dict], optional): Meta information of each sample.
    #             Defaults to None.
    #         gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
    #             Ground truth 3D boxes. Defaults to None.
    #         gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
    #             of 3D boxes. Defaults to None.
    #         gt_labels (list[torch.Tensor], optional): Ground truth labels
    #             of 2D boxes in images. Defaults to None.
    #         gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
    #             images. Defaults to None.
    #         img (torch.Tensor optional): Images of each sample with shape
    #             (N, C, H, W). Defaults to None.
    #         proposals ([list[torch.Tensor], optional): Predicted proposals
    #             used for training Fast RCNN. Defaults to None.
    #         gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
    #             2D boxes in images to be ignored. Defaults to None.

    #     Returns:
    #         dict: Losses of different branches.
    #     """
    #     pts_feats = self.extract_pts_feat(voxels, num_points, coors)
    #     losses = dict()
    #     losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
    #                                         gt_labels_3d, img_metas,
    #                                         gt_bboxes_ignore)
    #     losses.update(losses_pts)
    #     return losses

    # def forward_pts_train(self,
    #                       pts_feats,
    #                       gt_bboxes_3d,
    #                       gt_labels_3d,
    #                       img_metas,
    #                       gt_bboxes_ignore=None):
    #     """Forward function for point cloud branch.

    #     Args:
    #         pts_feats (list[torch.Tensor]): Features of point cloud branch
    #         gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
    #             boxes for each sample.
    #         gt_labels_3d (list[torch.Tensor]): Ground truth labels for
    #             boxes of each sampole
    #         img_metas (list[dict]): Meta information of samples.
    #         gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
    #             boxes to be ignored. Defaults to None.

    #     Returns:
    #         dict: Losses of each branch.
    #     """
    #     # outs = self.pts_bbox_head(pts_feats)
    #     loss_inputs = pts_feats + (gt_bboxes_3d, gt_labels_3d, img_metas)
    #     losses = self.pts_bbox_head.loss(
    #         *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
    #     return losses

    def simple_test(self, cls_scores, bbox_preds, dir_cls_preds, img_metas, rescale=False):
        """Test function of point cloud branch."""

        bbox_list = self.pts_bbox_head.get_bboxes(cls_scores, bbox_preds,
                                                  dir_cls_preds, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        bbox_res = [dict() for i in range(len(img_metas))]
        for result_dict, pts_bbox in zip(bbox_res, bbox_results):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_res

    # def simple_test(self, voxels=None, num_points=None, coors=None,
    #                 img_metas=None, img=None, rescale=False):
    #     """Test function without augmentaiton."
    #     pts_feats = self.extract_pts_feat(
    #         voxels, num_points, coors)

    #     bbox_list = [dict() for i in range(len(img_metas))]
    #     if pts_feats and self.with_pts_bbox:
    #         bbox_pts = self.simple_test_pts(
    #             pts_feats, img_metas, rescale=rescale)
    #         for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
    #             result_dict['pts_bbox'] = pts_bbox
    #     return bbox_list

    # def aug_test(self, points, img_metas, imgs=None, rescale=False):
    #     """Test function with augmentaiton."""
    #     img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)

    #     bbox_list = dict()
    #     if pts_feats and self.with_pts_bbox:
    #         bbox_pts = self.aug_test_pts(pts_feats, img_metas, rescale)
    #         bbox_list.update(pts_bbox=bbox_pts)
    #     return [bbox_list]

    # def extract_feats(self, points, img_metas, imgs=None):
    #     """Extract point and image features of multiple samples."""
    #     if imgs is None:
    #         imgs = [None] * len(img_metas)
    #     img_feats, pts_feats = multi_apply(self.extract_feat, points, imgs,
    #                                        img_metas)
    #     return img_feats, pts_feats

    # def aug_test_pts(self, feats, img_metas, rescale=False):
    #     """Test function of point cloud branch with augmentaiton."""
    #     # only support aug_test for one sample
    #     aug_bboxes = []
    #     for x, img_meta in zip(feats, img_metas):
    #         outs = self.pts_bbox_head(x)
    #         bbox_list = self.pts_bbox_head.get_bboxes(
    #             *outs, img_meta, rescale=rescale)
    #         bbox_list = [
    #             dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
    #             for bboxes, scores, labels in bbox_list
    #         ]
    #         aug_bboxes.append(bbox_list[0])

    #     # after merging, bboxes will be rescaled to the original image size
    #     merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
    #                                         self.pts_bbox_head.test_cfg)
    #     return merged_bboxes

    def show_results(self, data, result, out_dir):
        """Results visualization.

        Args:
            data (dict): Input points and the information of the sample.
            result (dict): Prediction results.
            out_dir (str): Output directory of visualization result.
        """
        for batch_id in range(len(result)):
            if isinstance(data['points'][0], DC):
                points = data['points'][0]._data[0][batch_id].numpy()
            elif mmcv.is_list_of(data['points'][0], torch.Tensor):
                points = data['points'][0][batch_id]
            else:
                ValueError(f"Unsupported data type {type(data['points'][0])} "
                           f'for visualization!')
            if isinstance(data['img_metas'][0], DC):
                pts_filename = data['img_metas'][0]._data[0][batch_id][
                    'pts_filename']
                box_mode_3d = data['img_metas'][0]._data[0][batch_id][
                    'box_mode_3d']
            elif mmcv.is_list_of(data['img_metas'][0], dict):
                pts_filename = data['img_metas'][0][batch_id]['pts_filename']
                box_mode_3d = data['img_metas'][0][batch_id]['box_mode_3d']
            else:
                ValueError(
                    f"Unsupported data type {type(data['img_metas'][0])} "
                    f'for visualization!')
            file_name = os.path.split(pts_filename)[-1].split('.')[0]

            assert out_dir is not None, 'Expect out_dir, got none.'
            inds = result[batch_id]['pts_bbox']['scores_3d'] > 0.1
            pred_bboxes = result[batch_id]['pts_bbox']['boxes_3d'][inds]

            # for now we convert points and bbox into depth mode
            if (box_mode_3d == Box3DMode.CAM) or (box_mode_3d
                                                  == Box3DMode.LIDAR):
                points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                                   Coord3DMode.DEPTH)
                pred_bboxes = Box3DMode.convert(pred_bboxes, box_mode_3d,
                                                Box3DMode.DEPTH)
            elif box_mode_3d != Box3DMode.DEPTH:
                ValueError(
                    f'Unsupported box_mode_3d {box_mode_3d} for convertion!')

            pred_bboxes = pred_bboxes.tensor.cpu().numpy()
            show_result(points, None, pred_bboxes, out_dir, file_name)


@DETECTORS.register_module()
class PointPillars(torch.nn.Module):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self,
                 save_onnx_path=None,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 #  pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 #  img_roi_head=None,
                 #  img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(PointPillars, self).__init__()

        # Save model for pointpillars.
        self.save_onnx_path = save_onnx_path

        if pts_voxel_layer:
            self.pts_voxel_layer = Pillar(**pts_voxel_layer)
        if pts_voxel_encoder:
            self.pts_voxel_encoder = builder.build_voxel_encoder(
                pts_voxel_encoder)
        if pts_middle_encoder:
            self.pts_middle_encoder = builder.build_middle_encoder(
                pts_middle_encoder)

        self.point_pillars_core = PointPillarsCore(
            pts_backbone=pts_backbone,
            pts_neck=pts_neck,
            pts_bbox_head=pts_bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        # if pts_bbox_head:
        #     pts_train_cfg = train_cfg.pts if train_cfg else None
        #     pts_bbox_head.update(train_cfg=pts_train_cfg)
        #     pts_test_cfg = test_cfg.pts if test_cfg else None
        #     pts_bbox_head.update(test_cfg=pts_test_cfg)
        #     self.pts_bbox_head = builder.build_head(pts_bbox_head)

        # self.train_cfg = train_cfg
        # self.test_cfg = test_cfg

        # if pretrained is None:
        #     pts_pretrained = None
        # elif isinstance(pretrained, dict):
        #     pts_pretrained = pretrained.get('pts', None)
        # else:
        #     raise ValueError(
        #         f'pretrained should be a dict, got {type(pretrained)}')
        # if self.with_pts_backbone:
        #     if pts_pretrained is not None:
        #         warnings.warn('DeprecationWarning: pretrained is a deprecated \
        #             key, please consider using init_cfg')
        #         self.pts_backbone.init_cfg = dict(
        #             type='Pretrained', checkpoint=pts_pretrained)

    def init_weights(self):
        self.point_pillars_core.init_weights()

    def train_step(self, data, optimizer):
        # print("train step:\n", data)
        losses = self(**data)
        loss, log_vars = self.point_pillars_core._parse_losses(losses)
        outputs = dict(loss=loss, log_vars=log_vars,
                       num_samples=len(data['img_metas']))
        return outputs

    def val_step(self, data, optimizer=None):
        losses = self(**data)
        loss, log_vars = self.point_pillars_core._parse_losses(losses)
        outputs = dict(loss=loss, log_vars=log_vars,
                       num_samples=len(data['img_metas']))
        return outputs

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxel_feats, coors = [], []
        # import pdb
        # pdb.set_trace()
        for res in points:
            res_voxels, res_coors = self.pts_voxel_layer(res)
            voxel_feats.append(res_voxels)
            coors.append(res_coors)
        voxels = torch.cat(voxel_feats, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, coors_batch

    def forward(self, points=None, return_loss=True, img_metas=None, img=None, **data):
        # print("forward data:\n", data)
        if return_loss:
            '''forward for train.'''
            data.update(
                {'img_metas': img_metas, 'img': img, 'return_loss': True})

        else:
            '''forward for test.'''
            for var, name in [(points, 'points'), (img_metas, 'img_metas')]:
                if not isinstance(var, list):
                    raise TypeError('{} must be a list, but got {}'.format(
                        name, type(var)))
            num_augs = len(points)
            if num_augs != len(img_metas):
                raise ValueError(
                    'num of augmentations ({}) != num of image meta ({})'.format(
                        len(points), len(img_metas)))

            if num_augs == 1:
                points = points[0]
                img = [img] if img is None else img
                data.update(
                    {'img_metas': img_metas[0], 'img': img[0], 'return_loss': False})
            else:
                # Now just annotated aug_test.
                raise NotImplementedError
                return self.aug_test(points, img_metas, img, **kwargs)

        # points size: torch.Size([58103, 4])
        pillars, coors = self.voxelize(points)
        # pillars size: [4512, 20, 10]. coors size: [4512, 4]
        pillar_features = self.pts_voxel_encoder(pillars)
        # voxel_features:  torch.Size([4512, 64])
        batch_size = coors[-1, 0] + 1
        # Scatter back to canvas.
        canvas_features = self.pts_middle_encoder(
            pillar_features, coors, batch_size)
        # x size: [1, 64, 468, 468]
        # PointPillars inference core: backbone, neck, head.
        cls_scores, bbox_preds, dir_cls_preds = self.point_pillars_core(
            canvas_features)
        # cls_scores[0].size: [1, 18, 468, 468], bbox_preds[0]: [1, 42, 468, 468], dir_cls_preds[0]: [1, 12, 468, 468]

        if self.save_onnx_path:
            pfe_name = os.path.join(self.save_onnx_path, "pfe.onnx")
            sim_pfe_name = os.path.join(self.save_onnx_path, "sim_pfe.onnx")
            rpn_name = os.path.join(self.save_onnx_path, "rpn.onnx")
            sim_rpn_name = os.path.join(self.save_onnx_path, "sim_rpn.onnx")

            # Save pfe model by onnx.
            # Dynamic input.
            input_names = ["pillars_input"]
            output_names = ["pillars_feats"]
            dynamic_axes = {
                "pillars_input": {0: "valid pillars"},
                "pillars_feats": {0: "valid pillars"}}
            torch.onnx.export(self.pts_voxel_encoder, (pillars,),
                              pfe_name, verbose=True, input_names=input_names,
                              output_names=output_names, dynamic_axes=dynamic_axes, opset_version=11)

            # Static input.
            # torch.onnx.export(self.pts_voxel_encoder, (pillars,),
            #                   pfe_name, verbose=True, input_names=input_names,
            #                   output_names=output_names, opset_version=11)
            print("Have saved pfe onnx model as: " + pfe_name)
            # check onnx model.
            import onnx
            import onnxsim
            onnx_model = onnx.load(pfe_name)
            # Dynamic input.
            model_sim, check = onnxsim.simplify(onnx_model, dynamic_input_shape=True,
                                                input_shapes={"pillars_input": [12000, 20, 10]})
            # Static input.
            ## model_sim, check = onnxsim.simplify(onnx_model)
            assert check, "Simplified pfe onnx model could not be validated"
            onnx.save(model_sim, sim_pfe_name)
            print("Have saved simplified pfe onnx model as: " + sim_pfe_name)

            # Save rpn model by onnx.
            input_names = ["canvas_features"]
            output_names = ["cls_scores", "bbox_preds", "dir_cls_preds"]
            torch.onnx.export(self.point_pillars_core, (canvas_features, ), rpn_name, verbose=True,
                              input_names=input_names, output_names=output_names, opset_version=11)
            onnx_model = onnx.load(rpn_name)
            model_sim, check = onnxsim.simplify(onnx_model)
            assert check, "Simplified rpn onnx model could not be validated"
            onnx.save(model_sim, sim_rpn_name)
            print("Have saved simplified rpn onnx model as: " + sim_rpn_name)
            exit()
        if return_loss:
            return self.forward_train(cls_scores, bbox_preds, dir_cls_preds, **data)
        else:
            return self.forward_test(cls_scores, bbox_preds, dir_cls_preds, **data)

    def forward_test(self, cls_score, bbox_pred, dir_cls_preds, img_metas=None, rescale=False, **kwargs):
        return self.point_pillars_core.simple_test(
            cls_score, bbox_pred, dir_cls_preds, img_metas, rescale=rescale)

    def forward_train(self, cls_score, bbox_pred, dir_cls_preds, gt_bboxes_3d,
                      gt_labels_3d,
                      img_metas,
                      gt_bboxes_ignore=None, **kwargs):
        final_losses = self.point_pillars_core.pts_bbox_head.loss(
            cls_score, bbox_pred, dir_cls_preds,
            gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore)
        return final_losses
