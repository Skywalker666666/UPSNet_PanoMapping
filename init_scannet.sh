##python init_scannet.py
##
##sed -i 's/semantic = np.zeros(pan.shape, dtype=np.uint8)/semantic = np.ones(pan.shape, dtype=np.uint8) * 255/g' lib/dataset_devkit/panopticapi/converters/panoptic2semantic_segmentation.py
##
##PYTHONPATH=$(pwd)/lib/dataset_devkit:$(pwd)/lib/dataset_devkit/panopticapi:$PYTHONPATH python lib/dataset_devkit/panopticapi/converters/panoptic2semantic_segmentation.py --input_json_file data/coco/annotations/panoptic_train2021_stff.json --segmentations_folder data/coco/annotations/panoptic_train2021 --semantic_seg_folder data/coco/annotations/panoptic_train2021_semantic_trainid_stff --categories_json_file data/coco/annotations/panoptic_scannet_categories_stff.json
##PYTHONPATH=$(pwd)/lib/dataset_devkit:$(pwd)/lib/dataset_devkit/panopticapi:$PYTHONPATH python lib/dataset_devkit/panopticapi/converters/panoptic2semantic_segmentation.py --input_json_file data/coco/annotations/panoptic_val2021_stff.json --segmentations_folder data/coco/annotations/panoptic_val2021 --semantic_seg_folder data/coco/annotations/panoptic_val2021_semantic_trainid_stff --categories_json_file data/coco/annotations/panoptic_scannet_categories_stff.json


# run this one first to get instance only file
PYTHONPATH=$(pwd)/lib/dataset_devkit:$(pwd)/lib/dataset_devkit/panopticapi:$PYTHONPATH python lib/dataset_devkit/panopticapi/converters/panoptic2detection_coco_format.py  --categories_json_file  /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/lib/dataset_devkit/panopticapi/panoptic_scannet_categories.json --input_json_file   /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/data/coco/annotations/panoptic_train2021.json --output_json_file /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/data/coco/annotations/instances_train2021.json --things_only


PYTHONPATH=$(pwd)/lib/dataset_devkit:$(pwd)/lib/dataset_devkit/panopticapi:$PYTHONPATH python lib/dataset_devkit/panopticapi/converters/panoptic2detection_coco_format.py  --categories_json_file  /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/lib/dataset_devkit/panopticapi/panoptic_scannet_categories.json --input_json_file   /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/data/coco/annotations/panoptic_val2021.json --output_json_file /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/data/coco/annotations/instances_val2021.json --things_only




# new one for debug: 03-19-2021
#PYTHONPATH=$(pwd)/lib/dataset_devkit:$(pwd)/lib/dataset_devkit/panopticapi:$PYTHONPATH python lib/dataset_devkit/panopticapi/converters/panoptic2detection_coco_format.py  --categories_json_file  /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/lib/dataset_devkit/panopticapi/panoptic_scannet_categories.json --input_json_file   /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/data/coco/annotations/panoptic_train2021.json --output_json_file /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/data/coco/annotations/instances_train2021_2.json --things_only

#PYTHONPATH=$(pwd)/lib/dataset_devkit:$(pwd)/lib/dataset_devkit/panopticapi:$PYTHONPATH python lib/dataset_devkit/panopticapi/converters/panoptic2detection_coco_format.py  --categories_json_file  /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/lib/dataset_devkit/panopticapi/panoptic_scannet_categories.json --input_json_file   /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/data/coco/annotations/panoptic_val2021.json --output_json_file /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/data/coco/annotations/instances_val2021_2.json --things_only

