import os
import json
import sys

class_list = ['aeroplane',   'bicycle',   'bird',   'boat',   'bottle',   'bus',   'car',   'cat',   'chair',   'cow',   'diningtable',   'dog',   'horse',   'motorbike',   'person',    'pottedplant',   'sheep',   'sofa',   'train',   'tvmonitor'] # VOC classes
#class_list = ['airplane',    'bicycle',   'bird',   'boat',   'bottle',   'bus',   'car',   'cat',   'chair',   'cow',   'dining table',  'dog',   'horse',   'motorbike',   'person',    'potted plant',  'sheep',   'couch',  'train',    'tv']  # COCO classes
class_list_coco = [name for name in json.load(open("/lab/tmpig8e/u/brian-data/COCO2017/VOC_COCO2017/label2id.json")).keys()]
pick_list =  {'aeroplane':50,'bicycle':50,'bird':50,'boat':50,'bottle':50,'bus':50,'car':50,'cat':50,'chair':50,'cow':50,'diningtable':50,'dog':50,'horse':50,'motorbike':50,'person':100,'pottedplant':50,'sheep':50,'sofa':50,'train':50,'tvmonitor':50}
step_list =  {'aeroplane':50,'bicycle':50,'bird':50,'boat':50,'bottle':50,'bus':50,'car':50,'cat':50,'chair':50,'cow':50,'diningtable':50,'dog':50,'horse':50,'motorbike':50,'person':100,'pottedplant':50,'sheep':50,'sofa':50,'train':50,'tvmonitor':50}
#threshold =  {'aeroplane':0.85, 'bicycle':0.85,'bird':0.85,'boat':0.90,'bottle':0.85,'bus':0.85,'car':0.85,'cat':0.85,'chair':0.85,'cow':0.85,'diningtable':0.5,'dog':0.85,'horse':0.85,'motorbike':0.85,'person':0.85,'pottedplant':0.85,'sheep':0.85,'sofa':0.85,'train':0.85,'tvmonitor':0.85}

option = "-W ignore::FutureWarning "
script = "select_mask_em_coco.py"
#save_dir = "/lab/tmpig8d/u/brian-data/VOCdevkit/VOC2012/COB_VOC_seg_split_mask/"
save_dir = "/lab/tmpig8e/u/brian-data/COCO2017/train2017_split_mask"
img_dir = "/lab/tmpig8e/u/brian-data/COCO2017/train2017_split/"
#mask_dir = "/lab/tmpig8d/u/brian-data/VOCdevkit/VOC2012/COB_VOC_seg_split/"  # output of COB seg
mask_dir = "/lab/tmpig8e/u/brian-data/COCO2017/train2017_split_entseg"  # output of entseg
extraction_dir = img_dir

num_machine = 8
if len(sys.argv) == 2:
    part = int(sys.argv[1])
    assert part != 0
    assert part <= num_machine
    class_list_coco = class_list_coco[int((part-1)*len(class_list_coco)/num_machine):int(part*len(class_list_coco)/num_machine)]
    print(class_list_coco)

for class_name in class_list_coco:
    #if class_name in ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike']:
    #    continue
    if class_name not in ["toaster","baseball bat","backpack","baseball glove","bowl","book","dining table","knife","mouse","skis","hair drier","handbag","snowboard","spoon","sports ball","truck","toothbrush"]:
        continue
    try:
        pick = str(pick_list[class_name])
    except:
        pick = "50"
    try:
        pick_step = str(step_list[class_name])
    except:
        pick_step = "50"
    #os.system("rm -rf " + os.path.join(save_dir,class_name))
    os.system("python "+option+script+" --img_root "+img_dir+" --mask_root "+mask_dir+" --extraction_root "+extraction_dir+" --save_root "+save_dir+" --num_iter 3 "+" --pick "+pick+" --pick_step "+pick_step+" --target_class \""+class_name+'\"')
