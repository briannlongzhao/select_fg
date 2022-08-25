import heapq
import json
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
import datetime
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

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s [%(levelname)8.8s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@dataclass
class Output:
    """
    H, W should be same (hard code as 299 in voc and 375 in coco)
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

class Images(OrderedDict):
    # Dict[str, Output]
    def filter_by(self, key_set: list) -> 'Images':
        return type(self)({
            k:v for k,v in self.items()
            if k in key_set
        })
    def all_best_segs(self) -> Iterator[np.ndarray]:
        return map(lambda o: o.best_seg, self.values())

class FEM:
    def __init__(self, args):
        self.args = args
        self.target_class: str
        # idx of corresponding target_class (eg index softmax pred by this)
        self.target_class_id: int
        self.label2id: Dict[str, int]
        self.model: Model
        self.img_size: int

    def prepare(
            self,
            dataset: str,
            batch_size: int,
            img_root: Path,
            mask_root: Path,
            target_class: str,
            feature_extractor: str = "xception",
            device="cuda:0"
    ) -> Images:
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
            self.img_size = 299
            with open("metadata/voc_label2id.json") as f:
                self.label2id = json.load(f)
            self.target_class_id = int(self.label2id[target_class])
            self.model = Model(model_name)  # Classifier
        elif dataset == "coco":
            # TODO: add coco classification model and configs
            self.img_size = 375
            raise NotImplementedError
            # self.model = PytorchModelWrapper_public('Q2L_COCO', 375, '/lab/tmpig8e/u/brian-data/COCO2017/VOC_COCO2017/label2id.json', 'cuda:0')
        else:
            raise NotImplementedError
        self.model = self.model.to(device).eval()
        self.model.register_gradcam(layer_name=self.model.find_target_layer(), img_size=self.img_size)
        self.feat_extractor = FeatureExtractor(feature_extractor, batch_size)
        self.feat_extractor = self.feat_extractor.to(device).eval()

        """
        prepare images
        images: Dict[imagename (eg 2007_000464.jpg) => Output]
            where Output = (image, seg mask, best seg (init None))
        """
        # images = OrderedDict()
        images = Images()
        for imgname in os.listdir(img_root / f"{self.target_class_id}_{target_class}"):
            imgname: str
            # imgname = imgname.split(".")[0] # no extension
            img_path = img_root / f"{self.target_class_id}_{target_class}" / imgname
            # (image_size, image_size, 3)
            img = Image.open(img_path).resize((self.img_size, self.img_size), Image.BILINEAR)
            img = np.array(img)
            mask_path = mask_root / f"{self.target_class_id}_{target_class}" / f"{imgname}.npy"
            assert mask_path.exists(), f"mask file {mask_path} must exist!!"
            # (H, W)
            # pixel value 1-#ent & 255
            entseg_mask = np.load(mask_path)
            # (self.image_size, self.image_size)
            entseg_mask = Image.fromarray(entseg_mask).resize((self.img_size, self.img_size), Image.NEAREST)
            entseg_mask = np.array(entseg_mask)
            images[imgname] = Output(img=img, mask=entseg_mask)
        return images

    def select_seg_by_gradcam(self, images: Images) -> Images:
        """
        step 1
        Given EntSeg, compute GradCAM for each image, then select best seg (only 1 for each image) based on GradCAM
        update @Images `best_seg` field
        """
        for imageid, output in tqdm(images.copy().items(), desc="selecting seg by GradCAM"):
            output: Output
            gradcam = self.model.compute_gradcam(output.img)
            best_seg, best_mask = self.best_seg_by_gradcam_center(output, gradcam, imageid)
            if best_seg is None:
                images.pop(imageid)
                continue
            images[imageid].best_seg = best_seg
            images[imageid].best_mask = best_mask
        return images

    def classifier_score(self, patch):
        """
        :param patch: RGB image that contains the object (after self.extract_scaleup_RGB!)
        :return: classifier's confidence score for @patch being @self.target_class_id
            higher := more likely to be the target class
        """
        preds, _ = self.model(patch)
        score = preds[0].cpu().numpy()[self.target_class_id].item()
        return score

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
    def best_seg_by_gradcam_center(self, output: Output, gradcam, imagename):
        """
        for all possible seg in @segmask, select best one
        :param output: contains
            img: (H, W, 3) original image
            mask: (H, W) mask, pixel value 1-#seg & 255 (dummy)
        :param gradcam: (H, W) GradCAM heatmap, binary
        :return: (masked image, mask)
            where mask is one of seg in @segmask, (H, W) binary, 1 = object
            and image is corresponding mask's region in @img but scaled up (H, W, 3) RGB int unnormalized
            NOTE: both None if no best seg is found
        """
        M = cv2.moments(gradcam, binaryImage=True)
        if M["m00"] == 0.0:
            M["m00"] = 0.001
        H, W = gradcam.shape  # should be always self.image_size, self.image_size
        cx = min(M["m01"] / M["m00"], H - 1)
        cy = min(M["m10"] / M["m00"], W - 1)

        """
        Euclidean distance for each ent seg
        """
        dists = []
        for m in output:
            # m: binary mask (H, W)
            mask_area = np.count_nonzero(m)
            if mask_area < 0.01 * m.size:
                continue
            if mask_area > 0.8 * m.size:
                continue
            # Maybe not necessary because dist not likely to select large bg
            all_ones = np.where(m == 1)  # tuple of 2d indices
            h1, h2, w1, w2 = all_ones[0].min(), all_ones[0].max(), all_ones[1].min(), all_ones[1].max()
            if len(m)-1-h2+h1 < 10 or len(m[0])-1-w2+w1 < 10:
                continue

            # avg of all pixel Euclidean distances to center of GradCAM
            dist = sum([
                distance.euclidean([x, y], [cx, cy])
                for (x, y), val in np.ndenumerate(m)
                if val == 1
            ]) / mask_area
            dists.append([dist, m])
        if len(dists) == 0:
            logger.warning("No proper segment in", imagename)
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
            patch = self.extract_scaleup_RGB(output.img, m)
            assert patch is not None, "patch shouldn't be malformed"
            score = self.classifier_score(patch)
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
               n_cluster: Optional[int] = None) -> Tuple[Images, np.ndarray]:
        """
        step 2: Calculate / Update mean of patch embeddings using k% closest to mean
        d: feature dimension (2048 in xception)
        N: number of images
        :return Images and also mean_emb
            if mode = mean: mean_emb (1, d)
            if mode is any gmm: mean_emb (n_cluster, d)
        """
        embs = self.feat_extractor.compute_embedding(images.all_best_segs())  # (N, d)
        if mode == "mean":
            assert metric != "mahalanobis"
            mean_emb = embs.mean(0, keepdims=True)  # (1, d)
            dist = distance.cdist(embs, mean_emb, metric=metric).squeeze()  # (N, )
        else:  # gmm
            """
            either use designated @n_cluster or, pick best n_cluster by lowest bic
            """
            covariance_type = "tied" if mode == "gmm_tied" else "full"
            if n_cluster is not None:
                n_components_range = [n_cluster]
            else:
                n_components_range = range(1, min(9, len(embs)))
            best_bic, best_gmm = float("inf"), None
            for n_component in n_components_range:
                # GMM param:
                #   means_ (n_cluster, d)
                #   covariances_ if full: (n_cluster, d, d); if tie: (d, d)
                gmm = mixture.GaussianMixture(n_components=n_component, random_state=42,
                                              covariance_type=covariance_type, reg_covar=1e-5)
                gmm = gmm.fit(embs)
                bic = gmm.bic(embs)
                if bic < best_bic:
                    best_bic, best_gmm = bic, gmm
            """
            compute distance
            1. use mahalanobis distance
                1.1. if gmm_tie: use shared covar (all clusters have same covar matrix (d, d))
                1.2. if gmm_full: each cluster has one covar, compute distance for each cluster
            2. else, use @metric
            """
            if metric == "mahalanobis":
                if mode == "gmm_tied":
                    # NOTE: share covar
                    inv_covar = np.linalg.pinv(gmm.covariances_)
                    # (N, n_cluster)
                    dist = distance.cdist(embs, gmm.means_, metric="mahalanobis", VI=inv_covar)
                else:
                    means, covars = best_gmm.means_, best_gmm.covariances_
                    # (N, n_cluster)
                    dist = np.zeros((len(embs), len(means))).astype("float")
                    for i_cluster, (mean, covar) in enumerate(zip(means, covars)):
                        inv_covar = np.linalg.pinv(covar)
                        dist[:, i_cluster] = distance.cdist(embs, mean[np.newaxis, :], metric="mahalanobis", VI=inv_covar).squeeze()
                dist = dist.min(axis=1)
            else:
                # (N, n_cluster)
                dist = distance.cdist(embs, best_gmm.means_, metric=metric)
                dist = dist.min(axis=1)
        """
        filter @images by k% closet to any of GMM cluster centroids
        closest def by @metric (default Euclidean Distance) or Mahalanobis Distance (if @use_mahalanobis)
        """
        top_k = int(k * len(embs))
        top_k_indices = np.argpartition(dist, top_k)[:top_k]
        images = Images({
            img: output
            for i, (img, output) in enumerate(images.items())
            if i in top_k_indices
        })
        if mode == "mean":
            embs = self.feat_extractor.compute_embedding(images.all_best_segs())
            mean_emb = embs.mean(0, keepdims=True)
        else: # gmm
            mean_emb = best_gmm.means_
        return images, mean_emb

    def E_step(self, orig_images: Images, mean_emb):
        """
        step 3: For each segment compute distance /similarity to the updated means and select the best
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
            emb = self.feat_extractor.compute_embedding(extract_image_gen(output))
            assert emb.shape[0] == len(possible_ent_segs)
            # (N, n_cluster)
            dist = distance.cdist(emb, mean_emb, metric="cosine")
            # (N, )
            dist = dist.min(1)
            best_idx = dist.argmin()
            output.best_mask, output.best_seg = possible_ent_segs[best_idx]

        return orig_images

    def save(self, images: Images, iter: int, output_dir: Path):
        """
        save intermediate results in @
        output directory =>
            <args.save_dir>
                <dataset>
                    <step-1>
                        step 1 results, ...
                        same structure as below
                    <M mode>[<M metric>,<M k>]
                        <target_class>
                            xx.png
                            yy.png
                        <target_class>_mask
                            xx.png
                            yy.png
        """
        if iter == -1:
            dir = output_dir.parent / "step-1"
        else:
            dir = output_dir / f"iter-{iter}"

        logger.info(f"saving in {dir}")
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

    def load_step1(self, images: Images, load_dir: Path):
        assert load_dir.exists(), f"with --load_step1, {load_dir} must exist!"
        logger.info(f"loading from {load_dir}")
        for imageid, output in images.items():
            output: Output
            imageid = imageid.replace("jpg", "png")
            seg_img, mask_img = load_dir / self.target_class / imageid, load_dir / f"{self.target_class}_mask" / imageid
            if seg_img.exists() and mask_img.exists():
                output.best_seg = np.array(Image.open(seg_img)).astype("uint8")
                output.best_mask = np.array(Image.open(mask_img)).astype("uint8") / 255
            else:
                logger.warning(f"{imageid} doesn't exist in {load_dir}, can be an error or because {imageid} is not valid")

        logger.info("loading done!")

    def run(self):
        """
        run F-EM algorithm
        """
        images: Images = self.prepare(
            dataset=self.args.dataset, img_root=self.args.img_root, mask_root=self.args.mask_root,
            target_class=self.args.target_class, batch_size=self.args.bsz,
            feature_extractor="xception", device="cuda:0")
        if self.args.debug:
            import random
            keep_keys = random.sample(images.keys(), int(len(images) * 0.1))
            logger.info(f"only keeping random 10% of {len(images)} input, i.e. {len(keep_keys)}")
            keep_keys = random.sample(images.keys(), int(len(images) * 0.1))
            images = images.filter_by(keep_keys)

        if self.args.load_step1:
            self.load_step1(images, self.args.save_root.parent / "step-1")
        else:
            images = self.select_seg_by_gradcam(images)
            self.save(images, iter=-1, output_dir=self.args.save_root)
        orig_images = images.copy()
        for iter in tqdm(range(self.args.num_iter), desc="EM iter"):
            _, mean_emb = self.M_step(
                images,
                mode=self.args.M_mode,
                k=self.args.M_k,
                metric=self.args.M_metric,
                n_cluster=self.args.M_n_cluster,
            )
            images = self.E_step(orig_images, mean_emb)
            self.save(images, iter, output_dir=self.args.save_root)
