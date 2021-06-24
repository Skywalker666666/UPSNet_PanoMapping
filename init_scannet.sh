# last updated at 06-21-2021
python init_scannet.py

# we might not need this one, I moved the processing to init_scannet.py
#sed -i 's/png", "height"/jpg", "height"/g' /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/data/coco/annotations/panoptic_train2021_stff.json
#sed -i 's/png", "height"/jpg", "height"/g' /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/data/coco/annotations/panoptic_val2021_stff.json

mv /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/data/coco/annotations/panoptic_train2021.json    /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/data/coco/annotations/panoptic_train2021_wrong_jpg_ext.json
mv /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/data/coco/annotations/panoptic_val2021.json    /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/data/coco/annotations/panoptic_val2021_wrong_jpg_ext.json

mv /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/data/coco/annotations/panoptic_train2021_jpg.json    /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/data/coco/annotations/panoptic_train2021.json
mv /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/data/coco/annotations/panoptic_val2021_jpg.json    /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/data/coco/annotations/panoptic_val2021.json


sed -i 's/semantic = np.zeros(pan.shape, dtype=np.uint8)/semantic = np.ones(pan.shape, dtype=np.uint8) * 255/g' lib/dataset_devkit/panopticapi/converters/panoptic2semantic_segmentation.py

PYTHONPATH=$(pwd)/lib/dataset_devkit:$(pwd)/lib/dataset_devkit/panopticapi:$PYTHONPATH python lib/dataset_devkit/panopticapi/converters/panoptic2semantic_segmentation.py --input_json_file data/coco/annotations/panoptic_train2021_stff.json --segmentations_folder data/coco/annotations/panoptic_train2021 --semantic_seg_folder data/coco/annotations/panoptic_train2021_semantic_trainid_stff --categories_json_file data/coco/annotations/panoptic_scannet_categories_stff.json
PYTHONPATH=$(pwd)/lib/dataset_devkit:$(pwd)/lib/dataset_devkit/panopticapi:$PYTHONPATH python lib/dataset_devkit/panopticapi/converters/panoptic2semantic_segmentation.py --input_json_file data/coco/annotations/panoptic_val2021_stff.json --segmentations_folder data/coco/annotations/panoptic_val2021 --semantic_seg_folder data/coco/annotations/panoptic_val2021_semantic_trainid_stff --categories_json_file data/coco/annotations/panoptic_scannet_categories_stff.json

# run this one first to get instance only file
# I run this one again at 06-21-2021
PYTHONPATH=$(pwd)/lib/dataset_devkit:$(pwd)/lib/dataset_devkit/panopticapi:$PYTHONPATH python lib/dataset_devkit/panopticapi/converters/panoptic2detection_coco_format.py  --categories_json_file  /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/lib/dataset_devkit/panopticapi/panoptic_scannet_categories.json --input_json_file   /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/data/coco/annotations/panoptic_train2021.json --output_json_file /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/data/coco/annotations/instances_train2021.json --things_only


PYTHONPATH=$(pwd)/lib/dataset_devkit:$(pwd)/lib/dataset_devkit/panopticapi:$PYTHONPATH python lib/dataset_devkit/panopticapi/converters/panoptic2detection_coco_format.py  --categories_json_file  /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/lib/dataset_devkit/panopticapi/panoptic_scannet_categories.json --input_json_file   /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/data/coco/annotations/panoptic_val2021.json --output_json_file /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/data/coco/annotations/instances_val2021.json --things_only





# new one for debug: 03-19-2021
#PYTHONPATH=$(pwd)/lib/dataset_devkit:$(pwd)/lib/dataset_devkit/panopticapi:$PYTHONPATH python lib/dataset_devkit/panopticapi/converters/panoptic2detection_coco_format.py  --categories_json_file  /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/lib/dataset_devkit/panopticapi/panoptic_scannet_categories.json --input_json_file   /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/data/coco/annotations/panoptic_train2021.json --output_json_file /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/data/coco/annotations/instances_train2021_2.json --things_only

#PYTHONPATH=$(pwd)/lib/dataset_devkit:$(pwd)/lib/dataset_devkit/panopticapi:$PYTHONPATH python lib/dataset_devkit/panopticapi/converters/panoptic2detection_coco_format.py  --categories_json_file  /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/lib/dataset_devkit/panopticapi/panoptic_scannet_categories.json --input_json_file   /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/data/coco/annotations/panoptic_val2021.json --output_json_file /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/data/coco/annotations/instances_val2021_2.json --things_only

# debug for polygon of sample image 03282021
#PYTHONPATH=$(pwd)/lib/dataset_devkit:$(pwd)/lib/dataset_devkit/panopticapi:$PYTHONPATH python lib/dataset_devkit/panopticapi/converters/panoptic2detection_coco_format.py  --categories_json_file  /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/lib/dataset_devkit/panopticapi/panoptic_scannet_categories.json --input_json_file   /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/data/coco/annotations/panoptic_train2025.json --output_json_file /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/data/coco/annotations/instances_train2025.json --things_only
#PYTHONPATH=$(pwd)/lib/dataset_devkit:$(pwd)/lib/dataset_devkit/panopticapi:$PYTHONPATH python lib/dataset_devkit/panopticapi/converters/panoptic2detection_coco_format.py  --categories_json_file  /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/lib/dataset_devkit/panopticapi/panoptic_scannet_categories.json --input_json_file   /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/data/coco/annotations/panoptic_train2025.json --output_json_file /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/data/coco/annotations/instances_train2025_bug_fix.json --things_only
