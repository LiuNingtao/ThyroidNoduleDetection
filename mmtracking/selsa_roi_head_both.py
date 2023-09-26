# Copyright (c) OpenMMLab. All rights reserved.
# mmtrack/models/roi_heads/selsa_roi_head_both.py
from mmdet.core import bbox2result, bbox2roi
from mmtrack.models import build_aggregator
from mmdet.models import HEADS, StandardRoIHead
import torch.nn as nn
import os
import torch
import numpy as np
from PIL import Image
from matplotlib.colors import hsv_to_rgb
import warnings
from PIL import Image

import sys

from globals import SharedData

@HEADS.register_module()
class AFPRoIHeadBoth(StandardRoIHead):
    """selsa roi head."""

    def __init__(self, aggregator, flow_generator=None,  *args, **kwargs):
        if flow_generator:
            sys.path.append(r'/srv/fenster/people/Ningtao/Project/USVideo/ARFlow/')
            from generate import TestHelper
            self.flow_generater = TestHelper(flow_generator)
        super().__init__(*args, **kwargs)
        self.aggregator = nn.ModuleList()
        for i in range(self.bbox_head.num_shared_fcs):
            self.aggregator.append(build_aggregator(aggregator))
        self.inplace_false_relu = nn.ReLU(inplace=False)
        self.fc_mask = nn.Linear(aggregator['in_channels'], 25088*4)

    def forward_train(self,
                      x,
                      ref_x,
                      img_metas,
                      proposal_list,
                      ref_proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            ref_x (list[Tensor]): list of multi-level ref_img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposal_list (list[Tensors]): list of region proposals.
            ref_proposal_list (list[Tensors]): list of region proposals
                from ref_imgs.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        if self.with_mask:
            ref_mask_result = self._mask_forward_train_ref(ref_x, ref_proposal_list)
            ref_mask_result = ref_mask_result['mask_pred']
        else:
            ref_mask_result = None
        # save_mask_hook(ref_mask_result)
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, ref_x, sampling_results,
                                                    ref_proposal_list,
                                                    ref_mask_result,
                                                    gt_bboxes, gt_labels, img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, ref_x, sampling_results,
                                                    ref_proposal_list,
                                                    ref_mask_result,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            # TODO: Support empty tensor input. #2280
            if mask_results['loss_mask'] is not None:
                 losses.update(mask_results['loss_mask'])

        return losses

    def aggregator_feature(self, x, ref_x, rois, ref_rois, ref_mask_result, mode):

        if mode == 'mask':
            N, C, H, W = x.shape

        if self.bbox_head.num_shared_convs > 0:
            for conv in self.bbox_head.shared_convs:
                x = conv(x)
                ref_x = conv(ref_x)

        if self.bbox_head.num_shared_fcs > 0:
            if self.bbox_head.with_avg_pool:
                x = self.bbox_head.avg_pool(x)
                ref_x = self.bbox_head.avg_pool(ref_x)
            
            x = x.flatten(1)
            ref_x = ref_x.flatten(1)

            for i, fc in enumerate(self.bbox_head.shared_fcs):
                x = fc(x)
                ref_x = fc(ref_x)
                if rois is not None and ref_rois is not None and len(ref_rois)>0:
                    x = x + self.aggregator[i](x, ref_x, rois, ref_rois, ref_mask_result)
                else:
                    print('SKIP')
                    x = x + self.aggregator[i](x, ref_x)
                ref_x = self.inplace_false_relu(ref_x)
                x = self.inplace_false_relu(x)
        if mode == 'mask':
            x = self.fc_mask(x)
            x = x.reshape((N, C, W*2, H*2))

        return x


    def _bbox_forward(self, x, ref_x, rois, ref_rois, ref_mask_result):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs],
            rois,
            ref_feats=ref_x[:self.bbox_roi_extractor.num_inputs])

        ref_bbox_feats = self.bbox_roi_extractor(
            ref_x[:self.bbox_roi_extractor.num_inputs], ref_rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
            ref_bbox_feats = self.shared_head(ref_bbox_feats)
        bbox_feats = self.aggregator_feature(bbox_feats, ref_bbox_feats, rois, ref_rois, ref_mask_result, mode='bbox')
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, ref_x, sampling_results,
                            ref_proposal_list, ref_mask_result, 
                            gt_bboxes, gt_labels, img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        ref_rois = bbox2roi(ref_proposal_list)
        bbox_results = self._bbox_forward(x, ref_x, rois, ref_rois, ref_mask_result)
        if self.train_cfg.get('save_at_train', False):
            bbox_results_formated = self._train_to_result(bbox_results, 
                                                            [res.bboxes for res in sampling_results],
                                                            img_metas,
                                                            self.test_cfg,
                                                        )
            try:
                top_bbox = bbox_results_formated[0][0][0].tolist()
                if top_bbox[-1] < 0.3:
                    top_bbox = None
            except:
                top_bbox = None
            if top_bbox:
                file_path = img_metas[0]['filename']
                image_dir, image_name = os.path.split(file_path)
                image_index = int(image_name.split('.')[0])
                path_pre = os.path.join(image_dir, format(image_index-1, '06d')+'.bmp')
                path_pre = file_path if not os.path.exists(path_pre) else path_pre
                path_nxt = os.path.join(image_dir, format(image_index+1, '06d')+'.bmp')
                path_nxt = file_path if not os.path.exists(path_nxt) else path_nxt
                flow_gray = self.get_flow_gray(file_path, path_pre, path_nxt, top_bbox)
            else:
                flow_gray = None
            SharedData.update_shared_var(img_metas[0]['ori_filename'], flow_gray)
            # globals.flow_gray_dict[img_metas[0]['ori_filename']] = float(flow_gray)
            # f = open(self.train_cfg['save_path'], mode='r')
            # save_dict = json.load(f)
            # f.close()
            # save_dict[img_metas[0]['ori_filename']] = float(flow_gray)
            # f = open(self.train_cfg['save_path'], mode='w')
            # json.dump(save_dict, f)
            
            # f.flush()
            # f.close()

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results
    
    def _mask_forward_train(self, x, ref_x, sampling_results, 
                            ref_proposal_list, ref_mask_result,
                            bbox_feats, gt_masks, img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if hasattr(self, 'share_roi_extractor') and not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            ref_rois = bbox2roi(ref_proposal_list)
            mask_results = self._mask_forward(x, ref_x, pos_rois, ref_rois, ref_mask_result)
        else:
            raise NotImplementedError

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, ref_x, rois=None, ref_rois=None, ref_mask_result=None):
        """Mask head forward function used in both training and testing."""
        mask_feats = self.mask_roi_extractor(
            x[:self.mask_roi_extractor.num_inputs], rois)
        ref_bbox_feats = self.mask_roi_extractor(
            ref_x[:self.mask_roi_extractor.num_inputs], ref_rois)
        if self.with_shared_head:
            mask_feats = self.shared_head(mask_feats)
            ref_bbox_feats = self.shared_head(ref_x)

       
        mask_feats = self.aggregator_feature(mask_feats, ref_bbox_feats, rois, ref_rois, ref_mask_result, mode='mask')
        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    def _mask_forward_train_ref(self, x, ref_proposal_list):
        """Run forward function and calculate loss for mask head in
        training."""
        if hasattr(self, 'share_roi_extractor') and not self.share_roi_extractor:
            pos_rois = bbox2roi(ref_proposal_list)
            mask_results = self._mask_forward_ref(x, pos_rois)
        else:
            mask_results = None
        return mask_results

    def _mask_forward_ref(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    def get_flow_gray(self, image_path, path_pre, path_nxt, det_bbox):
        def resize_flow(flow, new_shape):
            _, _, h, w = flow.shape
            new_h, new_w = new_shape
            flow = torch.nn.functional.interpolate(flow, (new_h, new_w),
                                                mode='bilinear', align_corners=True)
            scale_h, scale_w = h / float(new_h), w / float(new_w)
            flow[:, 0] /= scale_w
            flow[:, 1] /= scale_h
            return flow
        
        def flow_to_image(flow, max_flow=256):
            if max_flow is not None:
                max_flow = max(max_flow, 1.)
            else:
                max_flow = np.max(flow)

            n = 8
            u, v = flow[:, :, 0], flow[:, :, 1]
            mag = np.sqrt(np.square(u) + np.square(v))
            angle = np.arctan2(v, u)
            im_h = np.mod(angle / (2 * np.pi) + 1, 1)
            im_s = np.clip(mag * n / max_flow, a_min=0, a_max=1)
            im_v = np.clip(n - im_s, a_min=0, a_max=1)
            im = hsv_to_rgb(np.stack([im_h, im_s, im_v], 2))
            return (im * 255).astype(np.uint8)
        image = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
        h, w = image.shape[:2]
        mask = np.zeros_like(image)
        x1, y1, x2, y2 = list(map(lambda x: int(x), det_bbox[0:4]))
        mask[y1: y2, x1: x2, :] = 1
        image = image * mask
        image_pre = np.array(Image.open(path_pre).convert('RGB'), dtype=np.float32) * mask
        image_nxt = np.array(Image.open(path_nxt).convert('RGB'), dtype=np.float32) * mask
        flow_1 = self.flow_generater.run([image_pre, image])['flows_fw'][0].detach().cpu().numpy()
        flow_2 = self.flow_generater.run([image, image_nxt])['flows_fw'][0].detach().cpu().numpy()
        f_h, f_w = flow_1.shape[-2:]
        y1, y2 = int(y1 * f_h / h), int(y2 * f_h / h)
        x1, x2 = int(x1 * f_w / w), int(x2 * f_w / w)
        flow_1 = np.abs(flow_1[..., y1: y2, x1: x2])
        flow_2 = np.abs(flow_2[..., y1: y2, x1: x2])
        # flow_1 = flow_to_image(resize_flow(flow_1, (h, w))[0].detach().cpu().numpy().transpose([1, 2, 0]))
        # flow_2 = flow_to_image(resize_flow(flow_2, (h, w))[0].detach().cpu().numpy().transpose([1, 2, 0]))[y1: y2, x1: x2, :]
        # percentile_1 = np.percentile(flow_1, 5)
        # percentile_2 = np.percentile(flow_2, 5)
        mask_1 = flow_1 == flow_1
        mask_2 = flow_2 == flow_2
        gray_1 = np.ma.array(flow_1, mask=~mask_1).mean()
        gray_2 =  np.ma.array(flow_2, mask=~mask_2).mean()
        gray = (gray_1 + gray_2) / 2
        return gray
    
    def _train_to_result(self, bbox_results, proposals,
                         img_metas, rcnn_test_cfg, rescale=False):

        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = bbox2roi(proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        # some detector with_reg is False, bbox_pred will be None
        bbox_pred = bbox_pred.split(
            num_proposals_per_img,
            0) if bbox_pred is not None else [None, None]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]
        return bbox_results

    def simple_test(self,
                    x,
                    ref_x,
                    proposals_list,
                    ref_proposals_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        if self.with_mask:
            ref_mask_result = self._mask_forward_train_ref(ref_x, ref_proposals_list)
            ref_mask_result = ref_mask_result['mask_pred']
        else:
            ref_mask_result = None

        det_bboxes, det_labels = self.simple_test_bboxes(
            x,
            ref_x,
            proposals_list,
            ref_proposals_list,
            ref_mask_result,
            img_metas,
            self.test_cfg,
            rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            mask_results = self.simple_test_mask(
                x, ref_x, img_metas, det_bboxes, ref_proposals_list, ref_mask_result, det_labels, rescale=rescale)
            return list(zip(bbox_results, mask_results))

    def simple_test_mask(self,
                         x,
                         ref_x,
                         img_metas,
                         det_bboxes,
                         ref_proposals_list,
                         ref_mask_result,
                         det_labels,
                         rescale=False):
        """Simple test for mask head without augmentation."""
        # image shapes of images in the batch
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        if isinstance(scale_factors[0], float):
            warnings.warn(
                'Scale factor in img_metas should be a '
                'ndarray with shape (4,) '
                'arrange as (factor_w, factor_h, factor_w, factor_h), '
                'The scale_factor with float type has been deprecated. ')
            scale_factors = np.array([scale_factors] * 4, dtype=np.float32)

        num_imgs = len(det_bboxes)
        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            segm_results = [[[] for _ in range(self.mask_head.num_classes)]
                            for _ in range(num_imgs)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale:
                scale_factors = [
                    torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                    for scale_factor in scale_factors
                ]
            _bboxes = [
                det_bboxes[i][:, :4] *
                scale_factors[i] if rescale else det_bboxes[i][:, :4]
                for i in range(len(det_bboxes))
            ]
            mask_rois = bbox2roi(_bboxes)
            ref_rois = bbox2roi(ref_proposals_list)
            mask_results = self._mask_forward(x, ref_x, mask_rois, ref_rois, ref_mask_result)
            mask_pred = mask_results['mask_pred']
            # split batch mask prediction back to each image
            num_mask_roi_per_img = [len(det_bbox) for det_bbox in det_bboxes]
            mask_preds = mask_pred.split(num_mask_roi_per_img, 0)

            # apply mask post-processing to each image individually
            segm_results = []
            for i in range(num_imgs):
                if det_bboxes[i].shape[0] == 0:
                    segm_results.append(
                        [[] for _ in range(self.mask_head.num_classes)])
                else:
                    segm_result = self.mask_head.get_seg_masks(
                        mask_preds[i], _bboxes[i], det_labels[i],
                        self.test_cfg, ori_shapes[i], scale_factors[i],
                        rescale)
                    segm_results.append(segm_result)
        return segm_results

    def simple_test_bboxes(self,
                           x,
                           ref_x,
                           proposals,
                           ref_proposals,
                           ref_mask_result,
                           img_metas,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""


        rois = bbox2roi(proposals)
        ref_rois = bbox2roi(ref_proposals)
        bbox_results = self._bbox_forward(x, ref_x, rois, ref_rois, ref_mask_result)
        
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        # some detector with_reg is False, bbox_pred will be None
        bbox_pred = bbox_pred.split(
            num_proposals_per_img,
            0) if bbox_pred is not None else [None, None]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        
        if self.test_cfg.get('save_at_test', False):
            try:
                top_bbox = det_bboxes[0][0].tolist()
                if top_bbox[-1] < 0.3:
                    top_bbox = None
            except:
                top_bbox = None
            if top_bbox:
                file_path = img_metas[0]['filename']
                image_dir, image_name = os.path.split(file_path)
                image_index = int(image_name.split('.')[0])
                path_pre = os.path.join(image_dir, format(image_index-1, '06d')+'.bmp')
                path_pre = file_path if not os.path.exists(path_pre) else path_pre
                path_nxt = os.path.join(image_dir, format(image_index+1, '06d')+'.bmp')
                path_nxt = file_path if not os.path.exists(path_nxt) else path_nxt
                flow_gray = self.get_flow_gray(file_path, path_pre, path_nxt, top_bbox)
            else:
                flow_gray = None
            SharedData.update_shared_var(img_metas[0]['ori_filename'], flow_gray)
            # globals.flow_gray_dict[img_metas[0]['ori_filename']] = float(flow_gray)
            # f = open(self.train_cfg['save_path'], mode='r')
            # save_dict = json.load(f)
            # f.close()
            # save_dict[img_metas[0]['ori_filename']] = float(flow_gray)
            # f = open(self.train_cfg['save_path'], mode='w')
            # json.dump(save_dict, f)
            
            # f.flush()
            # f.close()


        return det_bboxes, det_labels


def save_mask_hook(mask_result: torch.Tensor):
    save_path = r'/path/to/mask_temp'
    mask_result = mask_result['mask_pred'].sigmoid().detach().cpu().numpy()
    for i in range(0, mask_result.shape[0], 50):
        # a = np.max(mask_result[i])
        # print(a)
        img_array = np.array((mask_result[i][0] > 0.5) * 255, dtype=np.uint8)
        Image.fromarray(img_array).convert('RGB').save(os.path.join(save_path, f"{str(i)}.png"))
