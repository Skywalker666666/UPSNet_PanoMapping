# ---------------------------------------------------------------------------
# Unified Panoptic Segmentation Network
#
# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Yuwen Xiong
# ---------------------------------------------------------------------------


import torch
from torch.autograd import Function
import numpy as np
from upsnet.bbox.sample_rois import sample_rois


class ProposalMaskTargetFunction(Function):

    def __init__(self, num_classes, batch_images, batch_rois, fg_fraction, mask_size, binary_thresh):
        super(ProposalMaskTargetFunction, self).__init__()
        self.num_classes = num_classes
        self.batch_images = batch_images
        self.batch_rois = batch_rois
        self.fg_fraction = fg_fraction
        self.mask_size = mask_size
        self.binary_thresh = binary_thresh

    def forward(self, rois, gt_boxes, gt_masks):

        context = torch.device('cuda', rois.get_device())

        assert self.batch_rois == -1 or self.batch_rois % self.batch_images == 0, \
            'batchimages {} must devide batch_rois {}'.format(self.batch_images, self.batch_rois)
        all_rois = rois.cpu().numpy()
        gt_boxes = gt_boxes.cpu().numpy()
        gt_masks = gt_masks.cpu().numpy()

        if self.batch_rois == -1:
            rois_per_image = all_rois.shape[0] + gt_boxes.shape[0]
            fg_rois_per_image = rois_per_image
        else:
            rois_per_image = self.batch_rois // self.batch_images
            fg_rois_per_image = np.round(self.fg_fraction * rois_per_image).astype(int)

        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack((all_rois, np.hstack((zeros, gt_boxes[:, :-1]))))
        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), 'Only single item batches are supported'

        rois, labels, bbox_targets, bbox_weights, mask_targets, mask_weights = \
            sample_rois(all_rois, fg_rois_per_image, rois_per_image, self.num_classes, gt_boxes=gt_boxes,
                        gt_masks=gt_masks, mask_size=self.mask_size, binary_thresh=self.binary_thresh)

        return torch.Tensor(rois).pin_memory().to(context, dtype=torch.float32, non_blocking=True), torch.Tensor(labels).pin_memory().to(context, dtype=torch.int64, non_blocking=True),\
               torch.Tensor(bbox_targets).pin_memory().to(context, dtype=torch.float32, non_blocking=True), torch.Tensor(bbox_weights).pin_memory().to(context, dtype=torch.float32, non_blocking=True),\
               torch.Tensor(mask_targets).pin_memory().to(context, dtype=torch.float32, non_blocking=True), torch.Tensor(mask_weights).pin_memory().to(context, dtype=torch.float32, non_blocking=True)

    def backward(self, grad_output):
        return None, None, None
