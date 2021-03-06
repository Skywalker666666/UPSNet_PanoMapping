import os
import sys
import torch
import torch.nn as nn
import argparse
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from upsnet.config.config import *
from upsnet.config.parse_args import parse_args
from upsnet.dataset.base_dataset import BaseDataset

from upsnet.models import *

from PIL import Image, ImageDraw

import copy

# *****************************************************************************************************
# global parameter
parser = argparse.ArgumentParser()

args, rest = parser.parse_known_args()

# for 1 gpu:
#args.cfg = "/home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/upsnet/experiments/upsnet_resnet50_coco_solo_1gpu.yaml"
#args.weight_path = "/home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/output/upsnet/coco/upsnet_resnet50_coco_1gpu/train2017/upsnet_resnet_50_coco_234000.pth"

# for 4 gpu or more:
args.cfg = "/home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/upsnet/experiments/upsnet_resnet50_coco_4gpu.yaml"
args.weight_path = "/home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/model/upsnet_resnet_50_coco_90000.pth"
# wrong
#args.weight_path = "/home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/model/upsnet_resnet_101_coco_pretrained_for_cityscapes.pth"


#solo_img_path = "/home/zhiliu/.mxnet/datasets/coco/val2017/"
#img_name = "000000022396.jpg"

solo_img_path = "/home/zhiliu/Documents/Datasets/SceneNN_more_sequence/scenenn_data/223/image/"
#img_name = "image04044.png"
img_name = "image00001.png"

args.eval_only = False

update_config(args.cfg)

# *****************************************************************************************************
# function borrowed from upsnet/dataset/base_data.py
def prep_im_for_blob(im, pixel_means, target_sizes, max_size):
    """Prepare an image for use as a network input blob. Specially:
      - Subtract per-channel pixel mean
      - Convert to float32
      - Rescale to each of the specified target size (capped at max_size)
    Returns a list of transformed images, one for each target size. Also returns
    the scale factors that were used to compute each returned image.
    """
    im = im.astype(np.float32, copy=False)
    if config.network.use_caffe_model:
        im -= pixel_means.reshape((1, 1, -1))
    else:
        im /= 255.0
        im -= np.array([[[0.485, 0.456, 0.406]]])
        im /= np.array([[[0.229, 0.224, 0.225]]])
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    print("im_shape: ")
    print(im_shape)
    print(im_shape[0:2])

    print("im_size_min: ")
    print(im_size_min)
    print("im_size_max: ")
    print(im_size_max)

    ims = []
    im_scales = []
    for target_size in target_sizes:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than max_size
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        print("im_scale: ")
        print(im_scale)
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        ims.append(im)
        im_scales.append(im_scale)
    return ims, im_scales




# function borrowed from upsnet/dataset/base_data.py
#def im_list_to_blob(ims, scale=1):
#    """Convert a list of images into a network input. Assumes images were
#    prepared using prep_im_for_blob or equivalent: i.e.
#      - BGR channel order
#      - pixel means subtracted
#      - resized to the desired input size
#      - float32 numpy ndarray format
#    Output is a 4D HCHW tensor of the images concatenated along axis 0 with
#    shape.
#    """
#    max_shape = np.array([im.shape for im in ims]).max(axis=0)
#    # Pad the image so they can be divisible by a stride
#    if config.network.has_fpn:
#        stride = float(config.network.rpn_feat_stride[-2])
#        max_shape[1] = int(np.ceil(max_shape[1] / stride) * stride)
#        max_shape[2] = int(np.ceil(max_shape[2] / stride) * stride)
#
#    num_images = len(ims)
#    blob = np.zeros((num_images, 3, int(max_shape[1] * scale), int(max_shape[2] * scale)),
#                    dtype=np.float32)
#    for i in range(num_images):
#        # transpose back to normal first, then after interpolate, transpose it back.
#        im = ims[i] if scale == 1 else cv2.resize(ims[i].transpose(1, 2, 0), None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
#        blob[i, :, 0:im.shape[1], 0:im.shape[2]] = im
#    # Move channels (axis 3) to axis 1
#    # Axis order will become: (batch elem, channel, height, width)
#    return blob


# function borrowed from upsnet/dataset/base_data.py
def get_unified_pan_result(segs, pans, cls_inds, stuff_area_limit=4 * 64 * 64):
    pred_pans_2ch = []

    for (seg, pan, cls_ind) in zip(segs, pans, cls_inds):
        pan_seg = pan.copy()
        pan_ins = pan.copy()
        id_last_stuff = config.dataset.num_seg_classes - config.dataset.num_classes
        ids = np.unique(pan)
        ids_ins = ids[ids > id_last_stuff]
        pan_ins[pan_ins <= id_last_stuff] = 0
        for idx, id in enumerate(ids_ins):
            region = (pan_ins == id)
            if id == 255:
                pan_seg[region] = 255
                pan_ins[region] = 0
                continue
            cls, cnt = np.unique(seg[region], return_counts=True)
            if cls[np.argmax(cnt)] == cls_ind[id - id_last_stuff - 1] + id_last_stuff:
                pan_seg[region] = cls_ind[id - id_last_stuff - 1] + id_last_stuff
                pan_ins[region] = idx + 1
            else:
                if np.max(cnt) / np.sum(cnt) >= 0.5 and cls[np.argmax(cnt)] <= id_last_stuff:
                    pan_seg[region] = cls[np.argmax(cnt)]
                    pan_ins[region] = 0
                else:
                    pan_seg[region] = cls_ind[id - id_last_stuff - 1] + id_last_stuff
                    pan_ins[region] = idx + 1

        idx_sem = np.unique(pan_seg)
        for i in range(idx_sem.shape[0]):
            if idx_sem[i] <= id_last_stuff:
                area = pan_seg == idx_sem[i]
                if (area).sum() < stuff_area_limit:
                    pan_seg[area] = 255

        print("pan.shape: ")
        print(pan.shape)
        pan_2ch = np.zeros((pan.shape[0], pan.shape[1], 3), dtype=np.uint8)

        pan_2ch[:, :, 0] = pan_seg
        pan_2ch[:, :, 1] = pan_ins
        pred_pans_2ch.append(pan_2ch)
    return pred_pans_2ch



# *****************************************************************************************************
# prepare the model
test_model = eval("resnet_50_upsnet")().cuda()
test_model.load_state_dict(torch.load(args.weight_path))

for p in test_model.parameters():
    p.requires_grad = False

test_model.eval()

# *****************************************************************************************************
# prepare input data
im = cv2.imread(os.path.join(solo_img_path, img_name))
print("size of raw input: ")
print(im.shape)
height = copy.deepcopy(im.shape[0])
print("height: " + str(height))
width  = copy.deepcopy(im.shape[1])
print("width: " + str(width))

# *****************************************************************************************************
# image resize, get scale
# method1
#im_resize = cv2.resize(im,(2048,1024),interpolation=cv2.INTER_CUBIC)
# method2
target_size = config.test.scales[0]
# default value   scales:800,  max_size: 1333
print("config.network.pixel_means: ")
print(config.network.pixel_means)
#print(config.train.max_size)
#print(target_size)

ims, im_scales = prep_im_for_blob(im, config.network.pixel_means, [target_size], config.test.max_size)

print("image size after resize: ")
print(ims[0].shape)

# *****************************************************************************************************
# image transpose, don't convert to torch tensor now
im_trans = ims[0].transpose(2, 0, 1)
#im_trans = torch.from_numpy(ims[0].transpose(2, 0, 1))
#im_trans = torch.unsqueeze(im_trans,0).type(torch.FloatTensor).cuda()

print("image size after transpose: ")
print(im_trans.shape)
im_infos = np.array([[
           ims[0].shape[0],
           ims[0].shape[1],
           im_scales[0]]], np.float32)



# *****************************************************************************************************
# batch collate
# prepare data fro fcn
batch = [{'data': im_trans , 'im_info' : im_infos}]
blob = {}
BD = BaseDataset()
for key in batch[0]:
    if key == 'data':
        blob.update({'data': torch.from_numpy(BD.im_list_to_blob([b['data'] for b in batch]))})
        if config.network.has_panoptic_head:
            blob.update({'data_4x': torch.from_numpy(BD.im_list_to_blob([b['data'] for b in batch], scale=1/4.))})
    elif key == 'im_info':
        blob.update({'im_info': np.vstack([b['im_info'] for b in batch])})

# *****************************************************************************************************
# batch cuda conversion
# TO DO, if we want to support multi GPU cards, we can add parallel information here
for k, v in blob.items():
    # 'data' and 'data_4x' are cuda type, img_info is non cuda
    blob[k] = v.cuda() if torch.is_tensor(v) else v

batch_clt = blob
#print(batch_clt)
print("shape of batch collation: ")
print(len(batch_clt))
print(batch_clt['data'].shape)
print(batch_clt['data_4x'].shape)


# *****************************************************************************************************
# run the prediction
#print(batch['im_info'])
with torch.no_grad():
    output_all = test_model(batch_clt)

output = {k: v.data.cpu().numpy() for k, v in output_all.items()} 



# *****************************************************************************************************
# coolect the output
sseg = output['fcn_outputs']
pano = output['panoptic_outputs']
pano_cls_ind = output['panoptic_cls_inds']

print("ssegs: ")
print(output['fcn_outputs'].shape)

print("shape of output['panoptic_outputs']: " )
print(output['panoptic_outputs'].shape)
print("unique labels in pano: ")
print(np.unique(output['panoptic_outputs']))

print("pano_class_ind: ")
print(output['panoptic_cls_inds'])



# *****************************************************************************************************
# restore the image to original size, and append it to a list, then can support both group and individual
ssegs = []
#for i, sseg in enumerate(output['ssegs']):
sseg = sseg.squeeze(0).astype('uint8')[:int(im_infos[0][0]), :int(im_infos[0][1])]
ssegs.append(cv2.resize(sseg, None, None, fx=1/im_infos[0][2], fy=1/im_infos[0][2], interpolation=cv2.INTER_NEAREST))

panos = []
pano = pano.squeeze(0).astype('uint8')[:int(im_infos[0][0]), :int(im_infos[0][1])]
#panos = cv2.resize(pano, None, None, fx=1/im_infos[0][2], fy=1/im_infos[0][2], interpolation=cv2.INTER_NEAREST)
panos.append(cv2.resize(pano, None, None, fx=1/im_infos[0][2], fy=1/im_infos[0][2], interpolation=cv2.INTER_NEAREST))
print("image infos: ")
print(im_infos[0][0])
print(im_infos[0][1])
print(im_infos[0][2])

pano_cls_inds = []
pano_cls_inds.append(pano_cls_ind)

print("ssegs[0] shape: ")
print(ssegs[0].shape)
#print(ssegs[0])

print("panos[0].shape: ")
print(panos[0].shape)
#print(panos)
#pano_segment = Image.fromarray(panox2[0])
#pano_segment.save("allnight.png")

# *****************************************************************************************************
# get final unified pan result
pred_pans_2ch = get_unified_pan_result(ssegs, panos, pano_cls_inds, stuff_area_limit=4 * 64 * 64)
#pan_2ch[:, :, 0] = pan_seg
#pan_2ch[:, :, 1] = pan_ins
print("unique label in final 2ch: ")
print("pan_seg: ")
print(np.unique(pred_pans_2ch[0][:,:,0]))
print("pan_ins: ")
print(np.unique(pred_pans_2ch[0][:,:,1]))


print(pred_pans_2ch[0][479,0:100,0])

pano_segment = Image.fromarray(pred_pans_2ch[0][:,:,0])
pano_instance = Image.fromarray(pred_pans_2ch[0][:,:,1]*255)

pano_segment.save("pano_segment.png")
pano_instance.save("pano_instance.png")

seg7_sub = (pred_pans_2ch[0][:,:,0]==7) * 255
seg7 = Image.fromarray(seg7_sub.astype('uint8'))
seg7.save("seg7.png")

seg51 = Image.fromarray(((pred_pans_2ch[0][:,:,0]==51)*255).astype('uint8'))
seg51.save("seg51.png")

seg42 = Image.fromarray(((pred_pans_2ch[0][:,:,0]==42)*255).astype('uint8'))
seg42.save("seg42.png")


seg255 = Image.fromarray(((pred_pans_2ch[0][:,:,0]==255)*255).astype('uint8'))
seg255.save("seg255.png")

