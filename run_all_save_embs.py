import json
import os
import sys

num_machine = 8

dataset = "coco"

class_list = [name for name in json.load(open(f"metadata/{dataset}_label2id.json")).keys()]

#skip_list = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car"]

script = "save_embs.py"
option = "-W ignore::FutureWarning "

if dataset == "voc":
    img_dir = "/lab/tmpig8d/u/brian-data/VOCdevkit/VOC2012/JPEGImages_split_multi/"
    mask_dir = "/lab/tmpig8d/u/brian-data/VOCdevkit/VOC2012/VOCmask_entseg/"
elif dataset == "coco":
    img_dir  = "/lab/tmpig8e/u/brian-data/COCO2017/train2017_split/"
    mask_dir = "/lab/tmpig8e/u/brian-data/COCO2017/train2017_split_entseg/"
save_dir = "/lab/tmpig8b/u/brian-data/VOCdevkit/1comp0.1/"

def run(run_list):
    for target_class in run_list:
        # if target_class != "cat":
        #     continue
        os.system(
            "python " + option + script +
            " --img_root " + img_dir +
            " --mask_root " + mask_dir +
            " --save_root " + save_dir +
            " --target_class " + '"' + target_class + '"' +
            " --dataset " + dataset +
            " --bsz " + str(4) +
            " --model " + "cls"
        )

if len(sys.argv) == 2:  # Split run on multiple machines
    part = int(sys.argv[1])
    assert part < num_machine, "index (0 based) larger than number of machines"
    class_list_part = class_list[int(part*len(class_list)/num_machine):int((part+1)*len(class_list)/num_machine)]
    print(class_list_part)
    run(class_list_part)
else:  # run all classes
    run(class_list)
