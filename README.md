# select_fg
## Description
Input the images and image segmentations generated by Entity Segmentation (split by class in subfolders), output the foreground segment of that class and its corresponding binary mask.

Below steps can be used to run the above file
1. Activate the environment `tsnecode` using below instruction: <br/>
    ` conda activate tsnecode`
2. The code is available in `/lab/andy/vibhav_code/Pytorch` directory.
    It can also be accessed from `~/vibhav_code/Pytorch`
3. The script can be run following below command: <br/>

ImagetNet:

```python 
python -W ignore::FutureWarning select_mask.py --img_root /lab/tmpig23c/u/andy/ILSVRC/Data/CLS-LOC/train/ --mask_root /lab/tmpig8d/u/zhix/Segement/Imagenet_mask/ --extraction_root /lab/tmpig23c/u/andy/ILSVRC/Data/CLS-LOC/train/ --save_root /lab/tmpig8c/u/brian_code/Pytorch/save_dir/ --num_iter 2 --pick 100 --pick_step 100 --target_class fire_engine
```

Pascal VOC:

```
python -W ignore::FutureWarning select_mask_em_pascalvoc.py --img_root /lab/tmpig8d/u/brian-data/VOCdevkit/VOC2012/JPEGImages_separate_singlelabel/ --mask_root /lab/tmpig8d/u/brian-data/VOCmask_entseg/ --extraction_root /lab/tmpig8d/u/brian-data/VOCdevkit/VOC2012/JPEGImages_separate_singlelabel/ --save_root /lab/tmpig8c/u/brian-code/Pytorch/save_dir/ --num_iter 2 --pick 50 --pick_step 50 --target_class aeroplane
```



Following is the description for each command-line arguments:

1. --img_root: 
   This argument holds the directory for original images. For example, /lab/tmpig23c/u/andy/ILSVRC/Data/CLS-LOC/train/ holds
   images for imagenet data
2. --mask_root:
   This argument holds directory for segmentation masks (in .npy format) generated from Entity Segmentation.
3. --extraction_root:
   This currently is same as img_root but can be modified if required
4. --save_root:
   This is the directory where all the outputs will be stored (details given below)
5. --num_iter: Number of iteration to run EM algorithm (default: 2)
6. --pick: How many images to pick in first iteration of EM algorithm (default: 200)
7. --pick_step: By how much to increment pick in every step
8. --target_class: target class for which we need to run the EM algorithm

Details for the output stored inside --save_root directory:
In the test_setup save_dir is at `/lab/andy/vibhav_code/Pytorch/save_dir` path
and below is the strucure:
```
/lab/andy/vibhav_code/Pytorch/save_dir/
├── fire_engine
└── school_bus
```

One directory down we have:
```
/lab/andy/vibhav_code/Pytorch/save_dir/fire_engine/
├── 500.pth
├── mean_dir #where all the means are saved with iter
├── pass_0 #pass {0} of the EM algorithm 
├── pass_1 
├── pass1_filtered #pass {1} filtered this can be used to verify result of first iter
├── pass1_filtered_orig_size #pass {1} original size 299x299
├── pass2_filtered 
└── pass2_filtered_orig_size
```

One can use the latest pass{int}_filtered output to see the output of the run 

