# Copyright (c) OpenMMLab. All rights reserved.
# mmtrack/models/aggregators/selsa_aggregator.py
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.core import BboxOverlaps2D

from ..builder import AGGREGATORS


@AGGREGATORS.register_module()
class SelsaAggregator(BaseModule):
    """Selsa aggregator module.

    This module is proposed in "Sequence Level Semantics Aggregation for Video
    Object Detection". `SELSA <https://arxiv.org/abs/1907.06390>`_.

    Args:
        in_channels (int): The number of channels of the features of
            proposal.
        num_attention_blocks (int): The number of attention blocks used in
            selsa aggregator module. Defaults to 16.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self, in_channels, num_attention_blocks=16, init_cfg=None):
        super(SelsaAggregator, self).__init__(init_cfg)
        self.fc_embed = nn.Linear(in_channels, in_channels)
        self.ref_fc_embed = nn.Linear(in_channels, in_channels)
        self.fc = nn.Linear(in_channels, in_channels)
        self.ref_fc = nn.Linear(in_channels, in_channels)
        self.num_attention_blocks = num_attention_blocks

    def forward(self, x, ref_x):
        """Aggregate the features `ref_x` of reference proposals.

        The aggregation mainly contains two steps:
        1. Use multi-head attention to computing the weight between `x` and
        `ref_x`.
        2. Use the normlized (i.e. softmax) weight to weightedly sum `ref_x`.

        Args:
            x (Tensor): of shape [N, C]. N is the number of key frame
                proposals.
            ref_x (Tensor): of shape [M, C]. M is the number of reference frame
                proposals.

        Returns:
            Tensor: The aggregated features of key frame proposals with shape
            [N, C].
        """
        roi_n, C = x.shape
        ref_roi_n, _ = ref_x.shape
        num_c_per_att_block = C // self.num_attention_blocks

        x_embed = self.fc_embed(x)
        # [num_attention_blocks, roi_n, C / num_attention_blocks]
        x_embed = x_embed.view(roi_n, self.num_attention_blocks,
                               num_c_per_att_block).permute(1, 0, 2)

        ref_x_embed = self.ref_fc_embed(ref_x)
        # [num_attention_blocks, C / num_attention_blocks, ref_roi_n]
        ref_x_embed = ref_x_embed.view(ref_roi_n, self.num_attention_blocks,
                                       num_c_per_att_block).permute(1, 2, 0)

        # [num_attention_blocks, roi_n, ref_roi_n]
        weights = torch.bmm(x_embed, ref_x_embed) / (x_embed.shape[-1]**0.5)
        weights = weights.softmax(dim=2)

        ref_x_new = self.ref_fc(ref_x)
        # [num_attention_blocks, ref_roi_n, C / num_attention_blocks]
        ref_x_new = ref_x_new.view(ref_roi_n, self.num_attention_blocks,
                                   num_c_per_att_block).permute(1, 0, 2)

        # [roi_n, num_attention_blocks, C / num_attention_blocks]
        x_new = torch.bmm(weights, ref_x_new).permute(1, 0, 2).contiguous()
        # [roi_n, C]
        x_new = self.fc(x_new.view(roi_n, C))
        return x_new

@AGGREGATORS.register_module()
class SelsaIoUAggregator(BaseModule):
    """Selsa aggregator module.

    This module is proposed in "Sequence Level Semantics Aggregation for Video
    Object Detection". `SELSA <https://arxiv.org/abs/1907.06390>`_.

    Args:
        in_channels (int): The number of channels of the features of
            proposal.
        num_attention_blocks (int): The number of at tention blocks used in
            selsa aggregator module. Defaults to 16.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self, 
                 in_channels, 
                 num_attention_blocks=16,
                 filter_component=None, 
                 aggre_component=None, 
                 aggre_factor=None, 
                 aux_factor=0.2,
                 mode=None,
                 init_cfg=None):
        super(SelsaIoUAggregator, self).__init__(init_cfg)
        self.fc_embed = nn.Linear(in_channels, in_channels)
        self.ref_fc_embed = nn.Linear(in_channels, in_channels)
        self.fc = nn.Linear(in_channels, in_channels)
        self.ref_fc = nn.Linear(in_channels, in_channels)
        self.num_attention_blocks = num_attention_blocks

        if filter_component:
            for _component in filter_component:
                assert _component in ['IoU', 'slope']  
                if _component == 'IoU':
                    self.IoU_calculator = BboxOverlaps2D()
            self.filter_component = filter_component
        else:
            self.filter_component = []

        self.aggre_factor = {}
        if aggre_component:
            if not aggre_factor:
                raise ValueError('if aggre_component provided, then aggre_factor is necessory')
            assert len(aggre_component) == len(aggre_factor)
            for idx, _component in enumerate(aggre_component):
                assert _component in ['length', 'mask', 'IoU']
                if _component == 'IoU':
                    self.IoU_calculator = BboxOverlaps2D()
                self.aggre_factor[_component] = aggre_factor[idx]
            self.aggre_component = aggre_component
        else:
            self.aggre_component = []
        if mode:
            assert mode in ['add', 'multiply'], f'Unsupported mode {mode}'
        self.mode = mode
        self.aux_factor = aux_factor

    def forward(self, x, ref_x, rois=None, ref_rois=None, ref_masks=None):
        """Aggregate the features `ref_x` of reference proposals.

        The aggregation mainly contains two steps:
        1. Use multi-head attention to computing the weight between `x` and
        `ref_x`.
        2. Use the normlized (i.e. softmax) weight to weightedly sum `ref_x`.

        Args:
            x (Tensor): of shape [N, C]. N is the number of key frame
                proposals.
            ref_x (Tensor): of shape [M, C]. M is the number of reference frame
                proposals.

        Returns:
            Tensor: The aggregated features of key frame proposals with shape
            [N, C].
        """
        roi_n, C = x.shape
        ref_roi_n, _ = ref_x.shape
        num_c_per_att_block = C // self.num_attention_blocks

        x_embed = self.fc_embed(x)
        # [num_attention_blocks, roi_n, C / num_attention_blocks]
        x_embed = x_embed.view(roi_n, self.num_attention_blocks,
                               num_c_per_att_block).permute(1, 0, 2)

        ref_x_embed = self.ref_fc_embed(ref_x)
        # [num_attention_blocks, C / num_attention_blocks, ref_roi_n]
        ref_x_embed = ref_x_embed.view(ref_roi_n, self.num_attention_blocks,
                                       num_c_per_att_block).permute(1, 2, 0)

        # [num_attention_blocks, roi_n, ref_roi_n]
        weights = torch.bmm(x_embed, ref_x_embed) / (x_embed.shape[-1]**0.5)
        # above is the basic part

        if ref_rois is not None and len(self.aggre_component) > 0:
            # weight block
            aggre_weights = []
            for _component in self.aggre_component:
                if 'length' == _component:
                    length = torch.sqrt((ref_rois[:, 1] - ref_rois[:, 3]) ** 2 + (ref_rois[:, 2] - ref_rois[:, 4]) ** 2)
                    length = (length - length.mean()) / length.std()
                    weight = length
                elif 'mask' == _component:
                    # ref_masks = ref_masks.sigmoid().detach()
                    # ref_masks = (ref_masks > 0.5) * 1.0
                    ref_masks = ref_masks.sum(dim=(1,2,3))
                    ref_masks = (ref_masks - ref_masks.mean()) / ref_masks.std()
                    weight = ref_masks
                elif 'IoU' == _component:
                    IoU_array = self.IoU_calculator(rois, ref_rois)
                    weight = IoU_array.mean(axis=0)
                weight = weight * self.aggre_factor[_component]
                aggre_weights.append(weight)
            aux_weights = torch.stack(aggre_weights, dim=1).sum(dim=1)
            aux_weights = (aux_weights - aux_weights.min()) / (aux_weights.max() - aux_weights.min())
            if self.mode == 'add':
                weights = (1-self.aux_factor) * weights + aux_weights[None, None, :] * self.aux_factor
            elif self.mode == 'multiply':
                weights = weights * aux_weights[None, None, :]
            
            # filter block
            filter_index = torch.ones(size=(roi_n, ref_roi_n), device=rois.device) == 0
            if len(rois) > 0:
                for _component in self.filter_component:
                    if _component == 'IoU':
                        IoU_array = self.IoU_calculator(rois, ref_rois)
                        # print(IoU_array.shape)
                        try:
                            filter_value = torch.quantile(IoU_array, 0.1)
                            filter_array = IoU_array > filter_value
                        except:
                            filter_array = IoU_array >= 0
                    elif _component == 'slope':
                        slope_roi = (rois[:, 1] - rois[:, 3]) / (rois[:, 2] - rois[:, 4])
                        slope_ref_roi = (ref_rois[:, 1] - ref_rois[:, 3]) / (ref_rois[:, 2] - ref_rois[:, 4])
                        slope_diff_array = torch.abs(slope_roi[:, None] - slope_ref_roi[None, :])
                        try:
                            filter_value = torch.quantile(slope_diff_array, 0.9)
                            filter_array = slope_diff_array < filter_value
                        except:
                            filter_array = slope_diff_array < 20
                    filter_index = filter_index & filter_array
            weights = weights * filter_index[None, :, :]                
        
        weights = weights.softmax(dim=2)
        ref_x_new = self.ref_fc(ref_x)
        # [num_attention_blocks, ref_roi_n, C / num_attention_blocks]
        ref_x_new = ref_x_new.view(ref_roi_n, self.num_attention_blocks,
                                   num_c_per_att_block).permute(1, 0, 2)

        # [roi_n, num_attention_blocks, C / num_attention_blocks]
        x_new = torch.bmm(weights, ref_x_new).permute(1, 0, 2).contiguous()
        # [roi_n, C]
        x_new = self.fc(x_new.view(roi_n, C))
        return x_new
