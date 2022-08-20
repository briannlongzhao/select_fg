import heapq
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Iterator, List

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial import distance
from sklearn import mixture
from tqdm.auto import tqdm

from model import Model, FeatureExtractor


@dataclass
class Output:
    """
    H, W should be same (hard code as 299)
    """
    # RGB (H, W, 3)
    img: np.ndarray
    # ent seg mask, pixel value 1-#ent & 255 (dummy)
    mask: np.ndarray
    # RGB of best ent (scaled up) (H, W, 3)
    best_seg: Optional[np.ndarray] = None
    # binary mask of best ent (H, W)
    best_mask: Optional[np.ndarray] = None

    def is_valid(self) -> bool:
        return self.best_seg is not None

    def all_ent_ids(self) -> List[int]:
        ent_ids = np.unique(self.mask).tolist()
        # ignore dummy
        return list(filter(lambda e: e != 255, ent_ids))

    def __iter__(self) -> Iterator[np.ndarray]:
        """
        iter all possible ent seg (w/o 255)
        each iter yield binary mask (1 = object), (H, W)
        """
        for entid in self.all_ent_ids():
            m = (self.mask == entid).astype(int)
            yield m


Images = Dict[str, Output]


class FEM:
    def __init__(self, args):
        self.args = args
        self.target_class: str
        # idx of corresponding target_class (eg index softmax pred by this)
        self.target_class_id: int
        self.label2id: Dict[str, int]
        self.model: Model

    def prepare(self, dataset: str, img_root: Path, mask_root: Path, target_class: str,
                feature_extractor: str = "xception", device="cuda:0") -> Images:
        """
        setup necessary model
        must be called first
        :param img_root: path of @dataset's original image
        :param mask_root: path of @dataset's ent seg
        :param feature_extractor: name of feature extractor, should be available in timm
        :return images, containing all images & classes
        """
        self.target_class = target_class
        if dataset == "voc":
            model_name = "resnet"
            img_size = 299
            with open(mask_root / "VOC_labels.json") as f:
                self.label2id = json.load(f)
            self.target_class_id = int(self.label2id[target_class])
        else:
            raise NotImplementedError
            # self.model = PytorchModelWrapper_public('Q2L_COCO', 375, '/lab/tmpig8e/u/brian-data/COCO2017/VOC_COCO2017/label2id.json', 'cuda:0')
        self.model = Model(model_name)
        self.model = self.model.to(device).eval()
        self.model.register_gradcam(layer_name=self.model.find_target_layer(), img_size=img_size)
        self.feat_extractor = FeatureExtractor(feature_extractor)
        self.feat_extractor = self.feat_extractor.to(device).eval()

        """
        prepare images
        images: Dict[imagename (eg 2007_000464.jpg) => Output]
            where Output = (image, seg mask, best seg (init None))
        """
        images = OrderedDict()
        for imgname in os.listdir(img_root / f"{self.target_class_id}_{target_class}"):
            imgname: str
            # imgname = imgname.split(".")[0] # no extension
            img_path = img_root / f"{self.target_class_id}_{target_class}" / imgname
            # (299, 299, 3)
            img = Image.open(img_path).resize((299, 299), Image.BILINEAR)
            img = np.array(img)
            mask_path = mask_root / f"{self.target_class_id}_{target_class}" / f"{imgname}.npy"
            assert mask_path.exists(), f"mask file {mask_path} must exist!!"
            # (H, W)
            # pixel value 1-#ent & 255
            entseg_mask = np.load(mask_path)
            # (299, 299)
            entseg_mask = Image.fromarray(entseg_mask).resize((299, 299), Image.NEAREST)
            entseg_mask = np.array(entseg_mask)
            images[imgname] = Output(img=img, mask=entseg_mask)
        return images

    def select_seg_by_gradcam(self, images: Images) -> Images:
        """
        step 1
        Given EntSeg, compute GradCAM for each image, then select best seg (only 1 for each image) based on GradCAM
        update @Images `best_seg` field
        """
        for imageid, output in tqdm(images.items(), desc="selecting seg by GradCAM"):
            output: Output
            gradcam = self.model.compute_gradcam(output.img)
            best_seg, best_mask = self.best_seg_by_gradcam_center(output, gradcam)
            if best_seg is None:
                continue
            images[imageid].best_seg = best_seg
            images[imageid].best_mask = best_mask
        return images

    @staticmethod
    def extract_scaleup_RGB(img, mask) -> Optional[np.ndarray]:
        """
        :param img: RGB (H, W, 3) unnormalized
        :param mask: binary (H, W)
        :return: RGB extracted image, scale up i.e. enlarge, (H, W, 3) unnormalized
            but if extract fail (eg area too small), return None
        """
        H, W = mask.shape
        mask_expanded = np.expand_dims(mask, -1)  # (H, W, 1)
        patch = mask_expanded * img + (1 - mask_expanded) * float(117)
        all_ones = np.where(mask == 1)  # tuple of 2d indices
        h1, h2, w1, w2 = all_ones[0].min(), all_ones[0].max(), all_ones[1].min(), all_ones[1].max()
        try:
            extracted_image = patch[h1:h2, w1:w2].astype(np.uint8)
            extracted_image = Image.fromarray(extracted_image)
        except:
            return
        extracted_image = extracted_image.resize((H, W), Image.BILINEAR)
        return np.array(extracted_image)

    @torch.no_grad()
    def best_seg_by_gradcam_center(self, output: Output, gradcam):
        """
        for all possible seg in @segmask, select best one
        :param output: contains
            img: (H, W, 3) riginal image
            mask: (H, W) mask, pixel value 1-#seg & 255 (dummy)
        :param gradcam: (H, W) GradCAM heatmap, binary
        :return: (masked image, mask)
            where mask is one of seg in @segmask, (H, W) binary, 1 = object
            and image is corresponding mask's region in @img but scaled up (H, W, 3) RGB int unnormalized
            NOTE: both None if no best seg is found
        """
        """
        gradcam center
        """
        M = cv2.moments(gradcam, binaryImage=True)
        if M["m00"] == 0.0:
            M["m00"] = 0.001
        H, W = gradcam.shape  # should be always 299, 299
        cx = min(M["m01"] / M["m00"], H - 1)
        cy = min(M["m10"] / M["m00"], W - 1)

        """
        Euclidean distance for each ent seg
        """
        dists = []
        for m in output:
            mask_area = np.count_nonzero(m)
            if mask_area < 0.01 * m.size:
                continue
            elif mask_area > 0.8 * m.size:
                continue
            all_ones = np.where(m == 1)  # tuple of 2d indices
            h1, h2, w1, w2 = all_ones[0].min(), all_ones[0].max(), all_ones[1].min(), all_ones[1].max()
            if len(m) - 1 - h2 + h1 < 10 or len(m[0]) - 1 - w2 + w1 < 10:
                continue

            # avg of all pixel Euclidean distance to center of GradCAM
            dist = sum([
                distance.euclidean([x, y], [cx, cy])
                for (x, y), val in np.ndenumerate(m)
                if val == 1
            ]) / mask_area
            dists.append([dist, m])
        if len(dists) == 0:
            # ignore this image
            return None, None

        """
        find best seg among top 3 closest (to gradcam center) seg
        by highest classifier score
        """
        heapq.heapify(dists)
        max_score = -float("inf")
        best_m = None
        for _ in range(min(3, len(dists))):
            _, m = heapq.heappop(dists)
            expanded_m = np.expand_dims(m, -1)  # (H, W, 1)
            patch = expanded_m * output.img + (1 - expanded_m) * float(117)
            preds, _ = self.model(patch)
            score = preds[0].cpu().numpy()[self.target_class_id].item()
            if score > max_score:
                max_score = score
                best_m = m
        """
        erosion best binary mask
        """
        best_m = cv2.erode(best_m.astype(np.uint8), np.ones((3, 3)), cv2.BORDER_REFLECT)
        best_image = self.extract_scaleup_RGB(output.img, best_m)
        return best_image, best_m

    def M_step(self, images: Images, k: float = 0.5, mode: str = "mean", metric: str = "euclidean",
               # only in mode=gmm
               n_cluster: int = 3, use_mahalanobis: bool = False) -> Tuple[Images, np.ndarray]:
        """
        step 2
        """
        # (N, d)
        embs = self.feat_extractor(map(lambda o: o.best_seg, images.values()))
        if mode == "mean":
            # (1, d)
            mean_emb = embs.mean(0, keepdims=True)
            # (N, )
            dist = distance.cdist(embs, mean_emb, metric=metric).squeeze()
        else: # gmm
            gmm = mixture.GaussianMixture(n_components=n_cluster, random_state=42, covariance_type="tied")
            # GMM param:
            #   means_ (n_cluster, d)
            #   covariances_ (d, d)
            gmm = gmm.fit(embs)
            if use_mahalanobis:
                # NOTE: share covar
                inv_covar = np.linalg.pinv(gmm.covariances_)
                # (N, n_cluster)
                dist = distance.cdist(embs, gmm.means_, metric="mahalanobis", VI=inv_covar)
                dist = dist.min(axis=1)
            else:
                # (N, n_cluster)
                dist = distance.cdist(embs, gmm.means_, metric=metric)
                dist = dist.min(axis=1)
        """
        filter @images by k% closet to any of GMM cluster centroids
        closest def by @metric (default Euclidean Distance) or Mahalanobis Distance (if @use_mahalanobis)
        """
        top_k = int(k * len(embs))
        top_k_indices = np.argpartition(dist, top_k)[:top_k]
        images = OrderedDict({
            img: output
            for i, (img, output) in enumerate(images.items())
            if i in top_k_indices
        })
        embs = self.feat_extractor(map(lambda o: o.best_seg, images.values()))
        mean_emb = embs.mean(0, keepdims=True)
        return images, mean_emb

    def E_step(self, orig_images: Images, mean_emb):
        """
        step 3
        """
        def extract_image_gen(output):
            nonlocal possible_ent_segs
            for m in output:
                extracted_image = self.extract_scaleup_RGB(output.img, m)
                if extracted_image is not None:
                    possible_ent_segs.append((m, extracted_image))
                    yield extracted_image

        for output in tqdm(orig_images.values(), desc="E step"):
            possible_ent_segs = []
            emb = self.feat_extractor(extract_image_gen(output))
            assert emb.shape[0] == len(possible_ent_segs)
            # (N, )
            dist = distance.cdist(emb, mean_emb, metric="cosine").squeeze()
            best_idx = dist.argmin()
            output.best_mask, output.best_seg = possible_ent_segs[best_idx]

        return orig_images

    def save(self, images: Images, iter: int, output_dir: Path):
        """
        save intermediate results in @
        output directory =>
            <output_dir>
                <target_class>
                    xx.png
                    yy.png
                <target_class>_mask
                    xx.png
                    yy.png
        """
        if iter == -1:
            subdir = "step-1"
        else:
            subdir = f"iter-{iter}"

        dir = output_dir / subdir
        print("saving in", dir)
        mask_dir = dir / f"{self.target_class}_mask"
        seg_dir = dir / self.target_class
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(seg_dir, exist_ok=True)
        for img, output in images.items():
            if output.is_valid():
                seg, mask = output.best_seg, output.best_mask
                seg = Image.fromarray(seg.astype(np.uint8))
                seg.save(seg_dir / img.replace("jpg", "png"))
                mask = Image.fromarray(mask.astype(np.uint8) * 255)
                mask.save(mask_dir / img.replace("jpg", "png"))

    def run(self):
        """
        run F-EM algorithm
        """
        images: Images = self.prepare(
            dataset=self.args.dataset, img_root=self.args.img_root, mask_root=self.args.mask_root,
            target_class=self.args.target_class,
            feature_extractor="xception", device="cuda:0")

        images = self.select_seg_by_gradcam(images)
        self.save(images, iter=-1, output_dir=self.args.save_root)
        orig_images = images.copy()
        for iter in tqdm(range(self.args.num_iter), desc="EM iter"):
            _, mean_emb = self.M_step(images,
                mode=self.args.M_mode, k=self.args.M_k, metric=self.args.M_metric,
                n_cluster=self.args.M_n_cluster, use_mahalanobis=self.args.M_mahalanobis)
            images = self.E_step(orig_images, mean_emb)
            self.save(images, iter, output_dir=self.args.save_root)