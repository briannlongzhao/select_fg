import json
import os
import numpy as np
import argparse
import sys
#sys.path.append('/lab/tmpig23/u/yao_code/Human-AI-Interface/')
# # from matplotlib import pyplot as plt
# # import matplotlib.gridspec as gridspec
# # import tsnecuda
# #import tcav.model as model
from PIL import Image
# #from skimage.segmentation import mark_boundaries
# #from sklearn import linear_model
from tqdm import tqdm
import torch
# #from torchvision import transforms
# #from sklearn.model_selection import cross_val_score
# #import tensorflow as tf
import cv2
import heapq
# from model_Pytorch import *
# from select_mask import *
# from gradcam_pytorch import GradCAM_model
import copy
import timm
import ace_helpers
from scipy import spatial
from sklearn import mixture
# import scipy.optimize as opt
import math
import warnings
warnings.filterwarnings("ignore")

dataset = "voc"

if dataset == "voc":
    sz = 299
    discover_class_dict = {
        'aeroplane': '0_aeroplane',
        'bicycle': '1_bicycle',
        'bird': '2_bird',
        'boat': '3_boat',
        'bottle': '4_bottle',
        'bus': '5_bus',
        'car': '6_car',
        'cat': '7_cat',
        'chair': '8_chair',
        'cow': '9_cow',
        'diningtable': '10_diningtable',
        'dog': '11_dog',
        'horse': '12_horse',
        'motorbike': '13_motorbike',
        'person': '14_person',
        'pottedplant': '15_pottedplant',
        'sheep': '16_sheep',
        'sofa': '17_sofa',
        'train': '18_train',
        'tvmonitor': '19_tvmonitor'
    } #modify as per the dataset
elif dataset == "coco":
    sz = 375
    with open("/lab/tmpig8e/u/brian-data/COCO2017/VOC_COCO2017/label2id.json") as f:  # label91
        label2id = json.load(f)
    with open("/lab/tmpig8e/u/brian-data/COCO2017/VOC_COCO2017/label80.txt") as f:  # label91
        i = 0
        for line in f.readlines():
            label2id[line.replace('\n', '')] = i
            i += 1

    discover_class_dict = {}
    for item in label2id.keys():
        discover_class_dict[item] = item
else:
    raise NotImplementedError

class SuperconceptDiscovery(object):

    def __init__(self, model, target_class, mask_dir, extraction_dir, test_img_dir, save_dir, discover_class_dict, device_, feat_ex):
        self.model = model #ace model
        self.feat_ex = feat_ex #feature extractor
        self.target_class = target_class #for which concept is to be generated
        self.mask_dir = mask_dir #Imagenet masks
        self.extraction_dir = extraction_dir #
        self.save_dir = save_dir
        self.test_img_dir = test_img_dir
        self.discover_class_dict = discover_class_dict
        self.device_ = device_
        self.filtered_indexes = []

    def load_concept_masks(self, target_class, mask_dir, img_dir):

        """load the mask for the target_class images under the given img_dir"""
        """TODO: Modify for VOC or make generic for all datatypes"""

        with open("/lab/tmpig8d/u/brian-data/VOCdevkit/VOC2012/pascalvoc_labels.json", 'r') as f:
            # json example: {"0": ["n01443537", "goldfish"]}
            # cats_dict: {"n01443537": (0, "goldfish")}
            # class_dict: {"goldfish": "n01443537"}
            cats = json.load(f)
            cats_dict = {}
            for k, v in cats.items():
                cats_dict[v[0]] = (int(k), v[1])

            class_dict = {}
            tmp_cats = cats.items()
            for k, v in tmp_cats:
                class_dict[v[1]] = v[0]

        concept_img_id = {}
        concept_img_id[target_class] = []
        target_masks_dir = os.path.join(mask_dir, self.discover_class_dict[target_class])
        image_dir = img_dir + self.discover_class_dict[target_class]
        #print(f"image dir {image_dir}")
        #print(f"total masks: {len(os.listdir(image_dir))}")
        for i in os.listdir(os.path.join(img_dir, self.discover_class_dict[target_class])):
            concept_img_id[target_class].append(i)

        target_masks = {}
        for mask_name in os.listdir(target_masks_dir):
            if ".npy" not in mask_name:
                continue
            img_id_name = mask_name.replace(".npy",'')
            if img_id_name in concept_img_id[target_class]:
                # load png mask
                #mask = np.array(Image.open(target_masks_dir + '/' + mask_name))
                # load npy mask
                mask = np.load(target_masks_dir + '/' + mask_name)
                target_masks[img_id_name] = mask
        return target_masks

    # Use pixel level average distance to the center of gradcam
    def extract_superpatch_center(self, img, image_mask, img_id):
        img, x, preds = self.model.get_preds(img) # returns: i/p, i/p to n/w, preds
        pre_class = preds[0].argmax()
        self.model.gradCAM_model.showCAMs(img, img_id, x=x.cuda(), chosen_class=pre_class, class_name=target_class, keep_percent=10)
        cam3, cam = self.model.gradCAM_model.compute_heatmap(x=x.cuda(), classIdx=pre_class, keep_percent=20)
        if dataset == "voc":
            target_class_idx = int(''.join(c for c in discover_class_dict[self.target_class] if c.isdigit()))
        elif dataset == "coco":
            target_class_idx = label2id[self.target_class]
        mask = copy.deepcopy(image_mask)
        mask_resized = np.array(Image.fromarray(mask).resize((sz,sz), Image.BILINEAR))
        min_dist = float('inf')
        seg_id = None
        heap = []
        # Find the center of Gaussian distribution
        '''def gauss2dFunc(xy, xo, yo, sigma):
            x = xy[0]
            y = xy[1]
            return np.exp(-((x - xo) * (x - xo) + (y - yo) * (y - yo)) / (2. * sigma * sigma)).ravel()
        xvec = np.array(range(sz))
        yvec = np.array(range(sz))
        X, Y = np.meshgrid(xvec, yvec)
        initial_guess = [150, 150, 50]  # x of center, y of center, variance
        try:
            popt, pcov = opt.curve_fit(gauss2dFunc, (X, Y), cam3[:,:,0].ravel(), p0=initial_guess)
            c1 = (popt[1],popt[0])
        except:
            c1 = (148,148)'''
        #print(img_id, end=' ')
        #print("c1:", round(c1[0]), round(c1[1]), end=' ')
        #Find center using moment
        c2 = self.model.gradCAM_model.find_gradcam_center(cam[:,:,0].astype('uint8'), sz)
        #print("c2:", round(c2[0]), round(c2[1]))

        if (img_id == "2010_000630.jpg"):
            found = True

        mask_selected = False
        # Calculate average distance to center
        def calc_dist(i, c):
            return math.sqrt((i[0]-c[0])**2+(i[1]-c[1])**2)
        for j in range(1, mask_resized.max()+1):
            m = copy.deepcopy(mask_resized)
            m[m != j] = 0
            m[m == j] = 1
            mask_area = np.count_nonzero(m)
            if mask_area < 0.01*m.size:
                continue
            if mask_area > 0.8*m.size:
                continue
            ones = np.where(m == 1)
            h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
            if len(m)-1-h2+h1 < 10 or len(m[0])-1-w2+w1 < 10:
                continue

            dist = (m.flatten() * [calc_dist(idx,c2) for idx,val in np.ndenumerate(m)]).sum() / mask_area
            heap.append((dist, m, j))
            '''if dist < min_dist:
                seg_id = j
                min_dist = dist
                # output_mask_img = mask_img.copy()
                output_mask = m.copy()
                mask_selected = True'''
        if len(heap) == 0:
            return None, None, None
        heapq.heapify(heap)
        max_score = -1 * float('inf')
        for i in range(3):
            try:
                _, m, mask_id = heapq.heappop(heap)
            except:
                break
            mask_expanded = np.expand_dims(m, -1)
            patch = (mask_expanded * img / 255 + (1 - mask_expanded) * float(117) / 255) * 255
            _, _, preds = self.model.get_preds(patch)
            score = (preds[0].cpu().numpy())[target_class_idx]
            if score > max_score:
                seg_id = mask_id
                output_mask = m.copy();
                max_score = score

        mask[mask!=seg_id] = 0
        mask[mask==seg_id] = 1
        # Erode noisy pixels
        output_mask = cv2.erode(output_mask, np.ones((3, 3)), cv2.BORDER_REFLECT)
        output_mask_original = cv2.erode(mask, np.ones((3,3)), cv2.BORDER_REFLECT)
        mask_expanded = np.expand_dims(output_mask, -1)
        patch = (mask_expanded * img/255 + (1 - mask_expanded) * float(117) / 255)
        ones = np.where(output_mask == 1)
        h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
        try:
            image = Image.fromarray((patch[h1:h2, w1:w2]*255).astype(np.uint8))
        except:
            try:
                image = Image.fromarray((patch[:, w1:w2] * 255).astype(np.uint8))
            except:
                try:
                    image = Image.fromarray((patch[h1:h2, :] * 255).astype(np.uint8))
                except:
                    image = Image.fromarray((patch * 255).astype(np.uint8))
        #if not os.path.exists(f"./images/exp/{self.target_class}/sp/"):
        #    os.makedirs(f"./images/exp/{self.target_class}/sp")
        #if not os.path.exists(save_dir+"gradcam_gaussian/"):
        #    os.makedirs(save_dir+"gradcam_gaussian/")
        #rp = image.resize((sz,sz),Image.BILINEAR)
        #rp.save(save_dir+"gradcam_gaussian/"+img_id.replace(".jpg", '')+'_'+str(seg_id)+".png")
        image_resized = np.array(image.resize((sz,sz), Image.BILINEAR)).astype(float)/255
        return image_resized, patch, output_mask_original

    def select_kcloser_to_mean(self, mean_idx, pick=200):
        mean_path = os.path.join(self.save_dir, self.target_class, "mean_dir")
        center = torch.load(ps.path.join(mean_path,'mean_'+str(mean_idx)+'.pth'))
        # print(f"center is {center[:5]}")
        dist_dic = {}
        target_masks = self.load_concept_masks(self.target_class, self.mask_dir, self.extraction_dir)
        cos_similarities = []
        for img_id, image_mask in target_masks.items():
            orig_img = np.array(Image.open(os.path.join(self.extraction_dir, self.discover_class_dict[self.target_class], img_id)).resize((sz,sz), Image.BILINEAR))
            mask = copy.deepcopy(image_mask)
            #need to change mask to use
            mask_resized = np.array(Image.fromarray(mask).resize((sz,sz), Image.BILINEAR))
            # print(f"mask resized: {mask_resized.shape}")
            # embedding, fg = self.get_embedding(img)
            cos_dic, mse_dic = self.get_all_cos(mask_resized, center, orig_img, img_id=img_id)
            # for seg in cos_dic.keys():
            #     print(f"seg is {seg}")
            #     print(cos_dic[seg])
            if len(cos_dic) > 0:
                min_cos = min(cos_dic.values(), key=lambda x:x[0])
                # print(f"min_cos: {min_cos[0]}")
                min_key = min(cos_dic, key=lambda x:cos_dic[x])
                # print(f"min_key: {min_key}, min_cos: {cos_dic[min_key][0]}")
                # print(max_cos == cos_dic[max_key])
                cos_similarities.append((min_cos[0], img_id, min_key))
        # heapq.heapify(cos_similarities)
        return heapq.nsmallest(pick, cos_similarities)

    def extract_superconcept(self,):
        '''extracts concept images and patches from utilizing gradcam and ace'''
        self.super_concept_images = []
        self.super_concept_patches = []
        self.super_concept_embed_dic = {}
        self.valid_indexes = {self.target_class:[]}
        target_masks = self.load_concept_masks(self.target_class, self.mask_dir, self.extraction_dir)
        if dataset == "voc":
            target_class_idx = int(''.join(c for c in discover_class_dict[self.target_class] if c.isdigit()))
        elif dataset == "coco":
            target_class_idx = label2id[self.target_class]
        result_path = os.path.join(concept.save_dir, concept.target_class)
        mask_path = os.path.join(concept.save_dir,concept.target_class+"_mask")
        os.makedirs(result_path, exist_ok=True)
        os.makedirs(mask_path, exist_ok=True)
        print("target_mask len: "+str(len(target_masks)))
        #mymodel = timm.create_model('xception', pretrained=True)
        #gradCAM_model = GradCAM_model(mymodel, sz, 'cuda:0', gradcam_layer='layer4.2.conv3')
        for img_id, image_mask in tqdm(target_masks.items()):
            img_original = np.array(Image.open(os.path.join(self.extraction_dir, self.discover_class_dict[self.target_class], img_id)))
            img = np.array(Image.open(os.path.join(self.extraction_dir, self.discover_class_dict[self.target_class], img_id)).resize((299,299), Image.BILINEAR))
            resized_patch, patch, mask = self.extract_superpatch_center(img, image_mask, img_id)
            if resized_patch is None:
                continue
            img_resized = resized_patch * 255
            # Use classifier to rule out unlikely concepts
            img = patch*255
            _, _, preds = self.model.get_preds(img)
            preds = (preds[0].cpu().numpy())
            if preds[target_class_idx] < 0.85:
                print(img_id, preds[target_class_idx], "skip")
                self.filtered_indexes.append()
                continue

            self.valid_indexes[self.target_class].append(img_id)
            self.super_concept_images.append((img_id, resized_patch))
            self.super_concept_patches.append(patch)
            img = Image.fromarray(img_resized.astype(np.uint8))
            img.save(os.path.join(result_path, img_id.replace("jpg", "png")))
            img_mask = Image.fromarray(mask.astype(np.uint8)*255)
            img_mask.save(os.path.join(mask_path, img_id.replace("jpg", "png")))


        class_save_dir = os.path.join(self.save_dir, self.target_class)
        if not os.path.exists(class_save_dir):
            os.makedirs(class_save_dir)
        # torch.save(self.super_concept_images, os.path.join(class_save_dir, '500.pth'))

        # debug
        '''Image.fromarray(patch).save(img_id)
        fig = plt.figure(figsize=(10,500))
        spec = fig.add_gridspec(ncols=1, nrows=300)
        for row in range(len(self.super_concept_images)):
            img = self.super_concept_images[row][1] #0 is image_id
            ax = fig.add_subplot(spec[row, 0])
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(str(row), color='w')
        fig.savefig(class_save_dir + '/' + '300.png')'''
        # end debug

        return self.valid_indexes

    def compute_mean(self, valid_indexes, mean_save_idx, in_image_path=None):
        '''computes mean for patch in layer4.2.conv3 feature space for valid indices'''
        self.super_concept_images = torch.load(os.path.join(self.save_dir, self.target_class, "500.pth"))
        valid_index = set()
        for index in valid_indexes[self.target_class]:
            if not isinstance(index, int) and isinstance(index, range):
                for i in index:
                    valid_index.add(i)
            else:
                valid_index.add(index)

        acts = []
        if in_image_path == None:
            for img_id, patch in self.super_concept_images:
                if img_id in valid_index:
                    # plt.imshow(patch)
                    # plt.show()
                    activation = self.model.run_examples(patch, 'layer4.2.conv3')
                    acts.append(np.mean(activation, (1,2)))
        else:
            for img_id in valid_indexes[self.target_class]:
                if not os.path.exists(in_image_path+ '/' + img_id.replace("jpg", "png")):
                    continue
                img = np.array(Image.open(os.path.join(in_image_path, img_id.replace("jpg", "png"))).resize((299,299), Image.BILINEAR)).astype(float)/255
                activation = self.model.run_examples(img, 'layer4.2.conv3')
                acts.append(np.mean(activation, (1,2)))

        acts = np.array(acts).reshape(len(acts), 2048)
        concept_mean = np.mean(acts, 0)
        # print(f"concept mean: {concept_mean[:5]}")
        class_save_dir = os.path.join(self.save_dir, self.target_class)
        if not os.path.exists(class_save_dir + '/mean_dir'):
            os.makedirs(os.path.join(class_save_dir, "mean_dir"))

        torch.save(concept_mean, os.path.join(class_save_dir, "mean_dir", "mean_"+str(mean_save_idx)+'.pth'))

    def get_all_cos(self, mask_resized, center, img, img_id=None):
        cos_dic = {}
        mse_dic = {}
        for j in range(1, mask_resized.max() + 1):
            m = copy.deepcopy(mask_resized)
            m[m != j] = 0
            m[m == j] = 1
            if np.count_nonzero(m) <= m.size / 100:
                continue
            output_mask = m.copy()
            mask_expanded = np.expand_dims(output_mask, -1)
            mask_img = (mask_expanded * img/255 + (1 - mask_expanded) * float(117) / 255)
            # print(f"get all cos")
            #just for visualization
            patch = copy.deepcopy(mask_img)
            if patch.shape != (sz, sz, 3):
                continue
            ones = np.where(output_mask == 1)
            h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
            try:
                image = Image.fromarray((patch[h1:h2, w1:w2]*255).astype(np.uint8))
            except:
                try:
                    image = Image.fromarray((patch[:, w1:w2] * 255).astype(np.uint8))
                except:
                    try:
                        image = Image.fromarray((patch[h1:h2, :] * 255).astype(np.uint8))
                    except:
                        image = Image.fromarray((patch * 255).astype(np.uint8))
            # plt.imshow(image)
            # plt.show()
            # image.save(f"./images/exp/verify/superkmeans/{img_id}_{j}.png")
            # mask_img = np.array(image.resize((sz,sz), Image.BILINEAR)).astype(float)/255
            # image.close()
            # im_mask = Image.fromarray(np.array(mask_img*255).astype(np.uint8)).resize((299, 299), Image.BILINEAR)
            # im_mask.save(f"./images/exp/verify/superkmeans/mask_{img_id}_{j}.png")
            # im_mask.close()

            # im_center = Image.fromarray(center).resize((sz, sz), Image.BILINEAR)
            # im_center.save("./images/exp/verify/superkmeans/center.png")
            # im_center.close()
            # print(f"get all cos ends")
            #mask_img = np.stack([m, m, m], axis=2)*img
            # print(f"mask img: {mask_img.shape}")
            if mask_img.shape != (sz,sz,3):
                continue
            seg_act = np.mean(mymodel.run_examples(mask_img, 'layer4.2.conv3'), (1,2)).squeeze(0)
            mse = np.sum((center-seg_act)**2)
            cos = spatial.distance.cosine(center, seg_act)
            # print(f"{img_id}: {j}: mse: {mse}, cos: {cos}")
            cos_dic[j] = [cos, center-seg_act]
            mse_dic[j] = [mse, center-seg_act]
        return cos_dic, mse_dic

def M_step(iter, target_class, res, discover_class_dict, extraction_dir, save_dir):
    vidx= {target_class:[]}
    seg = []
    n = len(res)
    for i in range(len(res)):
        vidx[target_class].append(res[i][1])
        seg.append(res[i][2])

    for i, img_id in tqdm(enumerate(vidx[target_class][:n])):
        img = np.array(Image.open(os.path.join(extraction_dir, discover_class_dict[target_class], img_id)).resize((299, 299), Image.BILINEAR))  # original image
        mask = copy.deepcopy(target_masks[img_id])
        mask_resized = np.array(Image.fromarray(mask).resize((sz, sz), Image.BILINEAR))
        m = copy.deepcopy(mask_resized)
        # print(f"seg: {seg[i]}")
        # for j in range(1, mask_resized.max() + 1):
        j = seg[i]
        m[m != j] = 0
        m[m == j] = 1
        if np.count_nonzero(m) <= m.size / 100:
            continue
        output_mask = m.copy()
        mask_expanded = np.expand_dims(output_mask, -1)
        patch = (mask_expanded * img / 255 + (1 - mask_expanded) * float(117) / 255)

        ones = np.where(m == 1)
        h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
        try:
            image = Image.fromarray((patch[h1:h2, w1:w2] * 255).astype(np.uint8))
        except:
            try:
                image = Image.fromarray((patch[:, w1:w2] * 255).astype(np.uint8))
            except:
                try:
                    image = Image.fromarray((patch[h1:h2, :] * 255).astype(np.uint8))
                except:
                    image = Image.fromarray((patch * 255).astype(np.uint8))
        # image_resized = np.array(image.resize((sz,sz), Image.BILINEAR)).astype(float)/255

        # out_img = image.resize((50,50), Image.BILINEAR)
        save_path = os.path.join(save_dir, target_class, 'pass_'+str(iter), str(n)+'_robust')
        if not os.path.exists(save_path):
            # os.makedirs(f"./images/exp/{target_class}/pass1/{n}_robust")
            os.makedirs(save_path)

        image.save(os.path.join(save_path, img_id.replace("jpg", "png")))
        image.close()

def E_step(iter, target_class, res, extraction_dir, save_dir):
    vidx = {target_class: []}
    seg = []
    for i in range(len(res)):
        vidx[target_class].append(res[i][1])
        seg.append(res[i][2])

    for i, img_id in tqdm(enumerate(vidx[target_class])):
        img = np.array(Image.open(os.path.join(extraction_dir, discover_class_dict[target_class], img_id)).resize((sz, sz),Image.BILINEAR))  # original image
        mask = copy.deepcopy(target_masks[img_id])
        mask_resized = np.array(Image.fromarray(mask).resize((sz, sz), Image.BILINEAR))
        m = copy.deepcopy(mask_resized)
        # print(f"seg: {seg[i]}")
        # for j in range(1, mask_resized.max() + 1):
        j = seg[i]
        m[m != j] = 0
        m[m == j] = 1
        if np.count_nonzero(m) <= m.size / 100:
            continue
        output_mask = m.copy()
        mask_expanded = np.expand_dims(output_mask, -1)
        patch = (mask_expanded * img / 255 + (1 - mask_expanded) * float(117) / 255)

        ones = np.where(m == 1)
        h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
        try:
            image = Image.fromarray((patch[h1:h2, w1:w2] * 255).astype(np.uint8))
        except:
            try:
                image = Image.fromarray((patch[:, w1:w2] * 255).astype(np.uint8))
            except:
                try:
                    image = Image.fromarray((patch[h1:h2, :] * 255).astype(np.uint8))
                except:
                    image = Image.fromarray((patch * 255).astype(np.uint8))

        save_orig_path = os.path.join(save_dir, target_class, "pass"+str(iter)+"_filtered_orig_size")
        if not os.path.exists(save_orig_path):
            # os.makedirs(f"./images/exp/{target_class}/pass1_filtered_orig_size")
            os.makedirs(save_orig_path)

        image.save(os.path.join(save_orig_path, img_id.replace("jpg", "png")))

        out_img = image.resize((50, 50), Image.BILINEAR)
        save_rs_path = os.path.join(save_dir, target_class, "pass"+str(iter)+"_filtered")
        if not os.path.exists(save_rs_path):
            # os.makedirs(f"./images/exp/{target_class}/pass1_filtered")
            os.makedirs(save_rs_path)

        out_img.save(os.path.join(save_rs_path, img_id.replace("jpg", "png")))
        out_img.close()
        image.close()

def parse_arguments(argv):
    """Parses the arguments passed to the run.py script."""
    parser = argparse.ArgumentParser()

    #Read folder data
    parser.add_argument('--img_root', type=str,
                      default='/lab/tmpig23c/u/andy/ILSVRC/Data/CLS-LOC/train/')  # original input images

    parser.add_argument('--mask_root', type=str,
                      default='/lab/tmpig8d/u/zhix/Segement/Imagenet_mask/') #masks from ES -> do we generate this

    parser.add_argument('--extraction_root', type=str,
                      default='/lab/tmpig23c/u/andy/ILSVRC/Data/CLS-LOC/train/')
    parser.add_argument('--save_root', type=str, default='/lab/andy/vibhav_code/Pytorch/save_dir/')
    parser.add_argument('--robust_img_dir', type=str,
                      default='/lab/andy/vibhav_code/Pytorch/images/exp') #Note: not used currently

    #Read parmeters
    # parser.add_argument('--selection_stage', type=str, choices=['gradCAM', 'E', 'M'])
    parser.add_argument('--num_iter', type=int, default=2)
    parser.add_argument('--pick', type=int, default=50) #select 200
    parser.add_argument('--pick_step', type=int, default=100)
    parser.add_argument('--ignore_small', type=float, default=0.01)
    parser.add_argument('--ignore_large', type=float, default=0.8)
    parser.add_argument('--classifier_thresh', type=float, default=0.5)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--target_class', type=str)
    parser.add_argument('--method', type=str, choices=["em","gmm"])

    return parser.parse_args(argv)


if __name__=='__main__':
    argv = parse_arguments(sys.argv[1:])
    if dataset == "voc":
        mymodel = ace_helpers.make_model('ResNet50', sz, '/lab/tmpig8d/u/brian-data/VOCdevkit/VOC2012/VOC_labels.json', 'cuda:1')
    elif dataset == "coco":
        mymodel = ace_helpers.make_model('Q2L_COCO', sz, '/lab/tmpig8e/u/brian-data/COCO2017/VOC_COCO2017/label2id.json', 'cuda:0')

    target_class = argv.target_class
    img_dir = argv.img_root
    mask_dir = argv.mask_root
    extraction_dir = argv.extraction_root
    save_dir = argv.save_root
    method = argv.method

    concept = SuperconceptDiscovery(mymodel, target_class, mask_dir, extraction_dir, img_dir, save_dir, discover_class_dict, 'cuda:0', None)

    all_valid_indexes = concept.extract_superconcept()  # extract valid img_ids

    if method is None:
        print("Done without EM")
        exit()


    if method == "gmm":
        #feat_ex = timm.create_model('xception', features_only=True, out_indices=[0,1,2,3,4], pretrained=True)
        feat_ex = timm.create_model('xception', pretrained=True)
        feat_extracted = {}
        def get_features(name):
            def hook(model, input, output):
                feat_extracted[name] = output.detach()
            return hook
        feat_ex.global_pool.register_forward_hook(get_features("feat"))
        target_masks = concept.load_concept_masks(target_class, mask_dir, extraction_dir)  # load all segmentation masks
        patches_tensor = torch.permute(torch.Tensor(concept.super_concept_patches), (0,3,1,2))
        output = feat_ex(patches_tensor).detach().numpy()
        feat_extracted = feat_extracted["feat"]
        #feat_extracted = torch.flatten(feat_ex(patches_tensor)[-1], start_dim=1).detach().numpy()
        lowest_bic = np.infty
        bic = []
        n_components_range = range(1, 9)
        cv_types = ["spherical", "tied", "diag", "full"]
        for n_components in n_components_range:
            for cv_type in cv_types:
                gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type)
                gmm.fit(feat_extracted)
                bic.append(gmm.bic(feat_extracted))
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    best_gmm = gmm
            print(n_components, cv_type)

        gmm_means = best_gmm.means_
        for images in concept.filtered_indexes:
            for i in range(1,256):



    exit()

    # EM
    target_masks = concept.load_concept_masks(target_class, mask_dir, extraction_dir)  # load masks
    #Compute mean for gradCAM
    concept.compute_mean(all_valid_indexes, 0)
    res = concept.select_kcloser_to_mean(0, pick=argv.pick)
    pick_c = argv.pick
    for iter in tqdm(range(argv.num_iter)):
        if iter != 0:
            img_path = os.path.join(save_dir, target_class, "/pass"+str(iter)+"_filtered_orig_size")
            concept.compute_mean(all_valid_indexes, iter+1, in_image_path=img_path)
            pick_c += argv.pick_step
            res = concept.select_kcloser_to_mean(iter+1, pick=pick_c)

        #M step
        M_step(iter, target_class, res, discover_class_dict, extraction_dir, save_dir)
        if argv.debug: #if debug true just verify if argv.pick is good to start with
            break

        #update mean
        n = len(res)
        robust_image_dir = os.path.join(save_dir, target_class, '/pass_'+str(iter), str(n)+'_robust')
        img_ids = [fname[:-4] for fname in os.listdir(robust_image_dir)]
        vidx_for_mean_update = {target_class: img_ids}
        concept.compute_mean(vidx_for_mean_update, iter+1, in_image_path=robust_image_dir)

        #E step
        res = concept.select_kcloser_to_mean(iter+1, pick=1300)
        E_step(iter+1, target_class, res, extraction_dir, save_dir)

    print("Done!")
