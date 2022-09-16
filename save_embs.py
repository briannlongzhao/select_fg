"""
Save embeddings of all extracted image segments in a pkl file.

pkl structure: {
    image_id_1: [
        [emb] or None,
        [emb] or None,
        ...
    ]
    image_id_2: [
        ...
    ]
    ...
}
"""

import pickle
import argparse
import shutil
import warnings
import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from FEM import FEM
from model import Model, FeatureExtractor
from FEM import Images, Output

warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_root', type=str, required=True, help="root directory that contains original images")
    parser.add_argument('--mask_root', type=str, required=True, help="root directory that contains .npy ent seg mask of each original image")
    parser.add_argument('--save_root', type=str, default="./embs/", help="where the embeddings are saved")

    parser.add_argument('--target_class', type=str, help="desired target class, must be in label2id (see metadata directory)")
    parser.add_argument('--model', type=str, choices=["featex", "cls"], default="featex", help="model to use for feature extraction")
    parser.add_argument('--dataset', type=str, default="voc", choices=["coco", "voc"], help="which dataest to run on")
    parser.add_argument('--bsz', type=int, default=16, help="batch size used in computing feature by feature extractor")
    parser.add_argument('--debug', action="store_true", default=False, help="if set, only run F-EM on subset of input")

    args = parser.parse_args()

    if args.model == "cls":
        if args.dataset == "voc":
            args.model_name = "resnet"
        elif args.dataset == "coco":
            args.model_name = "q2l"
        else:
            raise NotImplementedError
    elif args.model == "featex":
        args.model_name = "xception"
    else:
        raise NotImplementedError

    if args.dataset == "voc":
        args.img_size = 299
    elif args.dataset == "coco":
        args.img_size = 375
    else:
        raise NotImplementedError

    args.save_root = Path(args.save_root) / f"{args.dataset}" / "embs"
    args.mask_root = Path(args.mask_root)
    args.img_root = Path(args.img_root)
    return args

def prepare(img_root, mask_root, img_size, debug=False):
    images = Images()
    for imgname in tqdm(os.listdir(img_root)):
        img_path = img_root / imgname
        # (image_size, image_size, 3)
        img = Image.open(img_path).resize((img_size, img_size), Image.BILINEAR)
        img = np.array(img)
        mask_path = mask_root / f"{imgname}.npy"
        assert mask_path.exists(), f"mask file {mask_path} must exist!!"
        # (H, W)
        # pixel value 1-#ent & 255
        entseg_mask = np.load(mask_path)
        # (image_size, image_size)
        entseg_mask = Image.fromarray(entseg_mask)
        entseg_mask = np.array(entseg_mask)
        images[imgname] = Output(img=img, mask=entseg_mask)
        if debug:
            if len(images) == 100:
                break
    return images

if __name__ == '__main__':
    device = "cuda:0"
    args = parse_arguments()
    alg = FEM(args)
    alg.img_size = args.img_size

    model = FeatureExtractor(args.model_name, args.bsz) if args.model == "featex" else Model(args.model_name)
    model = model.to(device).eval()

    save_root = args.save_root / args.target_class
    img_root = args.img_root / args.target_class
    mask_root = args.mask_root / args.target_class
    os.makedirs(save_root, exist_ok=True)
    all_embs = {}
    images = prepare(img_root, mask_root, args.img_size, args.debug)
    for img_id, output in tqdm(images.items()):
        all_img_segs = []
        valid_ids = []
        embs = []
        for m, m_id in output:
            extracted_image, _ = alg.erode_mask_and_patch(output.img, m.copy())
            if extracted_image is None:
                continue
            else:
                valid_ids.append(m_id)
            all_img_segs.append(extracted_image)
        try:
            valid_embs = model.compute_embedding(all_img_segs)
            embs = [None for _ in range(max(valid_ids) + 1)]
        except Exception as e:
            print(img_id, e)
            all_embs[img_id] = None
            continue
        for seg_id, seg_emb in zip(valid_ids, valid_embs):
            embs[seg_id] = seg_emb
        all_embs[img_id] = embs

    save_root = save_root / f"{args.model_name}.pkl"
    if os.path.exists(save_root):
        os.remove(save_root)
    with open(save_root, 'wb') as f:
        pickle.dump(all_embs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("saved in", save_root)

    if args.debug:
        with open(save_root, 'rb') as f:
            embs = pickle.load(f)
            pass








