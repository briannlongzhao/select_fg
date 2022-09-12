import json
import os
import sys

num_machine = 10

dataset = "voc"

class_list = [name for name in json.load(open(f"metadata/{dataset}_label2id.json")).keys()]

#skip_list = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car"]

script = "select_mask.py"
option = "-W ignore::FutureWarning "

if dataset == "voc":
    img_dir = "/lab/tmpig8d/u/brian-data/VOCdevkit/VOC2012/JPEGImages_split_multi/"
    mask_dir = "/lab/tmpig8d/u/brian-data/VOCdevkit/VOC2012/VOCmask_entseg/"
elif dataset == "coco":
    img_dir  = "/lab/tmpig8e/u/brian-data/COCO2017/train2017_split/"
    mask_dir = "/lab/tmpig8e/u/brian-data/COCO2017/train2017_split_entseg/"
save_dir = "/lab/tmpig8b/u/brian-data/VOCdevkit/1comp0.1/"

M_mode = "gmm_full"
M_metric = "mahalanobis"
num_iter = 3
M_k = "0.3 0.5 0.7"
M_n_cluster = 1
filter_thresh = 0.1

def run(run_list):
    for target_class in run_list:
        # if target_class != "pottedplant":
        #     continue
        os.system(
            "python " + option + script +
            " --img_root " + img_dir +
            " --mask_root " + mask_dir +
            " --save_root " + save_dir +
            " --dataset " + dataset +
            " --num_iter " + str(num_iter) +
            " --bsz " + str(8) +
            " --M_mode " + M_mode +
            " --M_metric " + M_metric +
            " --M_n_cluster " + str(M_n_cluster) +
            " --M_k " + M_k +
            " --target_class " + '"' + target_class + '"' +
            #" --load_step1"
            " --filter_result" +
            " --filter_thresh " + str(filter_thresh)
        )

if len(sys.argv) == 2:  # Split run on multiple machines
    part = int(sys.argv[1])
    assert part < num_machine, "index (0 based) larger than number of machines"
    class_list_part = class_list[int(part*len(class_list)/num_machine):int((part+1)*len(class_list)/num_machine)]
    print(class_list_part)
    run(class_list_part)
else:  # run all classes
    run(class_list)
