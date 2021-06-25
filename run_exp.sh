 #python upsnet/upsnet_end2end_train.py --cfg upsnet/experiments/upsnet_resnet50_coco_1gpu.yaml 
#python upsnet/upsnet_end2end_test.py --cfg upsnet/experiments/upsnet_resnet50_coco_1gpu.yaml --weight_path output/upsnet/coco/upsnet_resnet50_coco_1gpu/train2017/upsnet_resnet_50_coco_234000.pth

#python upsnet/upsnet_end2end_test_single_img.py --cfg upsnet/experiments/upsnet_resnet50_coco_solo_1gpu.yaml --weight_path output/upsnet/coco/upsnet_resnet50_coco_1gpu/train2017/upsnet_resnet_50_coco_234000.pth


#python upsnet/core_upsnet_end2end_test_single_img.py --cfg upsnet/experiments/upsnet_resnet50_coco_1gpu.yaml --weight_path output/upsnet/coco/upsnet_resnet50_coco_1gpu/train2017/upsnet_resnet_50_coco_234000


#python upsnet/core_upsnet_end2end_test_multiple_imgs.py --cfg upsnet/experiments/upsnet_resnet50_coco_1gpu.yaml --weight_path output/upsnet/coco/upsnet_resnet50_coco_1gpu/train2017/upsnet_resnet_50_coco_234000   > running_log_muli_scenenn_02192021.txt



#args.cfg = "/home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/upsnet/experiments/upsnet_resnet50_coco_4gpu.yaml"
#args.weight_path = "/home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/model/upsnet_resnet_50_coco_90000.pth"


#python upsnet/upsnet_end2end_test.py --cfg upsnet/experiments/upsnet_resnet50_coco_1gpu.yaml --weight_path model/upsnet_resnet_50_coco_90000.pth





#python upsnet/core_upsnet_end2end_test_multiple_imgs.py --cfg upsnet/experiments/upsnet_resnet50_coco_4gpu.yaml --weight_path /home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/model/upsnet_resnet_50_coco_90000


# trainning script 03282021
#python upsnet/upsnet_end2end_train.py --cfg upsnet/experiments/upsnet_resnet50_scannet_1gpu.yaml 

#python upsnet/upsnet_end2end_test.py --cfg upsnet/experiments/upsnet_resnet50_scannet_1gpu.yaml --weight_path output/upsnet/scannet/upsnet_resnet50_scannet_1gpu/scannettrain2021/upsnet_resnet_50_scannet_126000.pth
# we renamed this result folder to: test_scannet_April2021_1st_successfule_test


# traininig and test 06222021
#python upsnet/upsnet_end2end_train.py --cfg upsnet/experiments/upsnet_resnet50_scannet_1gpu.yaml 

python upsnet/upsnet_end2end_test.py --cfg upsnet/experiments/upsnet_resnet50_scannet_1gpu.yaml --weight_path output/upsnet/scannet/upsnet_resnet50_scannet_1gpu/scannettrain2021/upsnet_resnet_50_scannet_130000.pth



