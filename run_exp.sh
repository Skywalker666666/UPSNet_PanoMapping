 #python upsnet/upsnet_end2end_train.py --cfg upsnet/experiments/upsnet_resnet50_coco_1gpu.yaml 


#python upsnet/upsnet_end2end_test.py --cfg upsnet/experiments/upsnet_resnet50_coco_1gpu.yaml --weight_path output/upsnet/coco/upsnet_resnet50_coco_1gpu/train2017/upsnet_resnet_50_coco_234000.pth

#python upsnet/upsnet_end2end_test_single_img.py --cfg upsnet/experiments/upsnet_resnet50_coco_solo_1gpu.yaml --weight_path output/upsnet/coco/upsnet_resnet50_coco_1gpu/train2017/upsnet_resnet_50_coco_234000.pth


python upsnet/core_upsnet_end2end_test_single_img.py --cfg upsnet/experiments/upsnet_resnet50_coco_1gpu.yaml --weight_path output/upsnet/coco/upsnet_resnet50_coco_1gpu/train2017/upsnet_resnet_50_coco_234000

