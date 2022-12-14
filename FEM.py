import heapq
import json
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Iterator, List

import cv2
import numpy as np
import torch
import pickle
from torch import nn
from PIL import Image
from scipy.spatial import distance
from sklearn import mixture
from tqdm.auto import tqdm

from model import Model, FeatureExtractor

logging.basicConfig(
    level=logging.INFO,
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
    # ent seg mask (original size), pixel value 1-#ent & 255 (dummy)
    mask: np.ndarray
    # Index of the best mask, value 1-#ent & 255 (dummy)
    best_seg_id: Optional[int] = -1
    # RGB of best ent (scaled up) (H, W, 3)
    best_seg: Optional[np.ndarray] = None
    # binary mask of best ent (original size)
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
            yield m, entid


class Images(OrderedDict):
    # Dict[str, Output]
    def filter_by(self, key_set: list) -> 'Images':
        return type(self)({
            k: v for k, v in self.items()
            if k in key_set
        })

    def all_best_segs(self) -> Iterator[np.ndarray]:
        return map(lambda o: o.best_seg, self.values())

    def all_best_segs_ids(self) -> Iterator[np.ndarray]:
        return map(lambda o: o.best_seg_id, self.values())

    def has_best_segs_ids(self) -> bool:
        return np.all(np.array(list(self.all_best_segs_ids())) > 0)

class FEM:
    def __init__(self, args):
        self.args = args
        self.target_class: str
        # idx of corresponding target_class (eg index softmax pred by this)
        self.target_class_id: int
        self.label2id: Dict[str, int]
        self.classifier: Model
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
        logger.info("preparing images")
        self.target_class = target_class
        if dataset == "voc":
            model_name = "resnet"
            self.img_size = 299
            with open("metadata/voc_label2id.json") as f:
                self.label2id = json.load(f)
            self.target_class_id = int(self.label2id[target_class])
        elif dataset == "coco":
            # TODO: add coco classification model and configs
            model_name = "q2l"
            self.img_size = 375
            with open("metadata/coco_label2id.json") as f:
                self.label2id = self.coco_label2id(json.load(f))
            self.target_class_id = int(self.label2id[target_class])
        else:
            raise NotImplementedError
        self.classifier = Model(model_name).to(device).eval()  # Classifier
        self.classifier.register_gradcam(layer_name=self.classifier.find_target_layer(), img_size=self.img_size)
        self.feat_extractor = FeatureExtractor(feature_extractor, batch_size)
        self.feat_extractor = self.feat_extractor.to(device).eval()

        """
        prepare images
        images: Dict[imagename (eg 2007_000464.jpg) => Output]
            where Output = (image, seg mask, best seg (init None))
        """
        images = Images()
        if self.args.dataset == "voc":
            cat_dir = f"{self.target_class_id}_{target_class}"
        elif self.args.dataset == "coco":
            cat_dir = target_class
        else:
            raise NotImplementedError
        for imgname in tqdm(os.listdir(img_root / cat_dir), desc="prepare"):
            imgname: str
            # imgname = imgname.split(".")[0] # no extension
            img_path = img_root / cat_dir / imgname
            # (image_size, image_size, 3)
            try:
                img = Image.open(img_path).resize((self.img_size, self.img_size), Image.BILINEAR)
            except:
                continue
            img = np.array(img)
            if np.max(img) == 0:
                continue
            mask_path = mask_root / cat_dir / f"{imgname}.npy"
            assert mask_path.exists(), f"mask file {mask_path} must exist!!"
            # (H, W)
            # pixel value 1-#ent & 255
            entseg_mask = np.load(mask_path)
            # (self.image_size, self.image_size)
            #entseg_mask = Image.fromarray(entseg_mask).resize((self.img_size, self.img_size), Image.NEAREST)
            entseg_mask = Image.fromarray(entseg_mask)
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
            try:
                gradcam = self.classifier.compute_gradcam(output.img)
            except Exception as e:
                print(imageid, e)
                continue
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
        preds, _ = self.classifier(patch)
        score = preds[0].detach().cpu().numpy()[self.target_class_id].item()
        return score

    def classifier_score_coco(self, saved_embs, image_id, m_id):
        """
        :param saved_embs: mask id of segment
        :return: classifier's confidence score for @patch being @self.target_class_id
            higher := more likely to be the target class
        """
        assert saved_embs is not None
        emb = self.retrieve_embedding(saved_embs, [image_id], [[m_id]])
        if emb is None:
            return None
        cls_score = torch.squeeze(torch.Tensor(emb)).softmax(-1)
        assert len(cls_score) == 80
        return cls_score[self.target_class_id]

    def extract_scaleup_RGB(self, img, mask) -> Optional[np.ndarray]:
        """
        :param img: RGB (H, W, 3) unnormalized
        :param mask: binary (original size)
        :return: RGB extracted image, scale up i.e. enlarge, (H, W, 3) unnormalized
            but if extract fail (eg area too small), return None
        """
        mask = self.resize_mask(mask)
        mask_expanded = np.expand_dims(mask, -1)  # (H, W, 1)
        patch = mask_expanded * img + (1 - mask_expanded) * float(117)
        all_ones = np.where(mask == 1)  # tuple of 2d indices
        try:
            h1, h2, w1, w2 = all_ones[0].min(), all_ones[0].max(), all_ones[1].min(), all_ones[1].max()
            extracted_image = patch[h1:h2, w1:w2].astype(np.uint8)
            extracted_image = Image.fromarray(extracted_image)
        except:
            return
        extracted_image = extracted_image.resize((self.img_size, self.img_size), Image.BILINEAR)
        return np.array(extracted_image)

    def erode_mask_and_patch(self, image, mask):
        mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3)), cv2.BORDER_REFLECT)
        patch = self.extract_scaleup_RGB(image, mask)
        return patch, mask

    def resize_mask(self, mask):
        mask_resized = Image.fromarray(mask.astype(np.uint8)).resize((self.img_size, self.img_size), Image.NEAREST)
        mask_resized = np.array(mask_resized)
        return mask_resized

    @staticmethod
    def coco_label2id(lab2id91):
        lab2id80 = {}
        i = 0
        for k,v in lab2id91.items():
            lab2id80[k] = i
            i += 1
        return lab2id80

    @staticmethod
    def is_feasible_mask(m):
        # m: binary mask (H, W)
        mask_area = np.count_nonzero(m)
        if mask_area < 0.01 * m.size:
            return False
        # Maybe not necessary because dist not likely to select large bg
        all_ones = np.where(m == 1)  # tuple of 2d indices
        h1, h2, w1, w2 = all_ones[0].min(), all_ones[0].max(), all_ones[1].min(), all_ones[1].max()
        if len(m) - 1 - h2 + h1 < 5 or len(m[0]) - 1 - w2 + w1 < 5:
            return False
        return True

    @torch.no_grad()
    def best_seg_by_gradcam_center(self, output: Output, gradcam, imagename):
        """
        for all possible seg in @segmask, select best one
        :param output: contains
            img: (H, W, 3) original image
            mask: (original size) mask, pixel value 1-#seg & 255 (dummy)
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
        for m, _ in output:
            m_resized = self.resize_mask(m)
            if not self.is_feasible_mask(m_resized):
                continue

            # avg of all pixel Euclidean distances to center of GradCAM
            dist = sum([
                distance.euclidean([x, y], [cx, cy])
                for (x, y), val in np.ndenumerate(m_resized)
                if val == 1
            ]) / np.count_nonzero(m_resized)
            dists.append([dist, m])
        if len(dists) == 0:
            logger.warning(f"No proper segment in {imagename}")
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
            # expanded_m = np.expand_dims(m, -1)  # (H, W, 1)  # No resize
            # patch = expanded_m * output.img + (1 - expanded_m) * float(117)  # No resize
            patch = self.extract_scaleup_RGB(output.img, m)  # Resize
            assert patch is not None, "patch shouldn't be malformed"
            score = self.classifier_score(patch)
            if score > max_score:
                max_score = score
                best_m = m
        """
        erosion best binary mask
        """
        return self.erode_mask_and_patch(output.img, best_m.copy())

    def retrieve_embedding(self, saved_embs, image_ids, available_ids, mode='M'):
        """
        For all image_id retrieve saved embeddings of available_ids (2d array)
        Return ndarray of size num_segs by feat_dim
        """
        assert len(image_ids) == len(available_ids)
        embs = []
        for img_id, seg_ids in zip(image_ids, available_ids):
            for m_id in seg_ids:
                if saved_embs[img_id] is None:
                    if mode == 'M':
                        continue
                    if mode == 'E':
                        return None
                embs.append(saved_embs[img_id][m_id])
        embs = np.array(embs)
        return embs

    def M_step(
        self,
        images: Images,
        feat_extractor: nn.Module = None,
        k: float = 0.5,
        mode: str = "mean",
        metric: str = "euclidean",
        filter_thresh: float = -1.0,
        saved_embs=None,
        # only in mode=gmm
        n_cluster: Optional[int] = None
    ) -> Tuple[Images, np.ndarray]:
        """
        step 2: Calculate / Update mean of patch embeddings using k% closest to mean
        d: feature dimension (2048 in xception)
        N: number of images
        :return Images and also mean_emb
            if mode = mean: mean_emb (1, d)
            if mode is any gmm: mean_emb (n_cluster, d)
        """
        feat_extractor = self.feat_extractor if feat_extractor is None else feat_extractor
        if filter_thresh > 0:
            filtered_ids = [
                id for id, output in images.items()
                if output.is_valid()
                and self.classifier_score(output.best_seg) > filter_thresh
            ]
            logger.info(f"filter: {len(filtered_ids)}/{len(images)} images to compute mean")
            assert len(filtered_ids) > 0
            images = images.filter_by(filtered_ids)

        if saved_embs is not None and images.has_best_segs_ids():
            embs = self.retrieve_embedding(saved_embs, images.keys(), [[id] for id in list(images.all_best_segs_ids())])
        else:
            embs = feat_extractor.compute_embedding(images.all_best_segs())  # (N, d)

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
                try:
                    gmm = mixture.GaussianMixture(n_components=n_component, random_state=42, covariance_type=covariance_type, reg_covar=1e-4)
                except:
                    continue
                gmm = gmm.fit(embs)
                bic = gmm.bic(embs)
                if bic < best_bic:
                    best_bic, best_gmm = bic, gmm
            assert best_gmm is not None, "no fit gmm"
            logger.info(f"best n_component={best_gmm.n_components}")
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
        logger.info(f"top {k}: {len(top_k_indices)}/{len(embs)} images to compute mean")
        if mode == "mean":
            if saved_embs is not None and images.has_best_segs_ids():
                embs = self.retrieve_embedding(saved_embs, images.keys(), [[id] for id in list(images.all_best_segs_ids())])
            else:
                embs = feat_extractor.compute_embedding(images.all_best_segs())
            mean_emb = embs.mean(0, keepdims=True)
        else: # gmm
            mean_emb = best_gmm.means_
        return images, mean_emb

    def E_step(
        self,
        orig_images: Images,
        mean_emb,
        feat_extractor: nn.Module = None,
        saved_embs=None,
    ):
        """
        step 3: For each segment compute distance/similarity to the updated means and select the best
        """
        def extract_image_gen(output):
            nonlocal possible_ent_segs
            for m, m_id in output:
                if self.is_feasible_mask(m):
                    extracted_image, _ = self.erode_mask_and_patch(output.img, m.copy())
                    if extracted_image is not None:
                        possible_ent_segs.append((m, m_id, extracted_image))
                        yield extracted_image
        feat_extractor = self.feat_extractor if feat_extractor is None else feat_extractor
        for image_id, output in tqdm(orig_images.items(), desc="E step"):
            possible_ent_segs = []
            if saved_embs is None:
                emb = feat_extractor.compute_embedding(extract_image_gen(output))
            else:
                _ = list(extract_image_gen(output))
                emb = self.retrieve_embedding(saved_embs, [image_id], [[x[1] for x in possible_ent_segs]], mode='E')
            if emb is None or emb.shape[0] == 0:
                continue
            # (N, n_cluster)
            try:
                dist = distance.cdist(emb, mean_emb, metric="cosine")
            except Exception as e:
                print(image_id, e)
                continue
            # (N, )
            dist = dist.min(1)
            best_idx = dist.argmin()
            output.best_mask, output.best_seg_id, output.best_seg = possible_ent_segs[best_idx]

        return orig_images

    def save(self, images: Images, iter: int, output_dir: Path, filter_thresh=-1.0):
        """
        save intermediate results in @
        output directory =>
            <args.save_dir>
                <dataset>
                    <step-1>
                        step 1 results, ...
                        same structure as below
                    <saved_embs>
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
                if self.args.filter_thresh > 0 and self.classifier_score(seg) < filter_thresh:
                    continue
                seg = Image.fromarray(seg.astype(np.uint8))
                seg.save(seg_dir / img.replace("jpg", "png"))
                mask = Image.fromarray(mask.astype(np.uint8) * 255)
                mask.save(mask_dir / img.replace("jpg", "png"))

    def load_step1(self, images: Images, load_dir: Path):
        assert load_dir.exists(), f"with --load_step1, {load_dir} must exist!"
        logger.info(f"loading from {load_dir}")
        for imageid, output in tqdm(images.items(), desc="load_step1"):
            output: Output
            imageid = imageid.replace("jpg", "png")
            seg_img, mask_img = load_dir / self.target_class / imageid, load_dir / f"{self.target_class}_mask" / imageid
            if seg_img.exists() and mask_img.exists():
                output.best_seg = np.array(Image.open(seg_img)).astype("uint8")
                output.best_mask = np.array(Image.open(mask_img)).astype("uint8") / 255
            else:
                logger.warning(f"{imageid} doesn't exist in {load_dir}, can be an error or because {imageid} is not valid")

        logger.info("loading done!")

    def load_embeddings(self, load_dir: Path):
        assert load_dir.exists(), f"with --load_embs, {load_dir} must exist!"
        logger.info(f"loading embeddings from {load_dir}")
        with open(load_dir, 'rb') as f:
            return pickle.load(f)


    def post_process(self, images1: Images, images2: Images):
        """
        Select output with higher score from 2 sets of proposals
        """
        images = Images()
        assert len(images1) == len(images2), "proposals should have save number of images"
        for (image1, output1), (image2, output2) in tqdm(zip(images1.items(), images2.items()), desc="post_process"):
            assert image1 == image2
            try:
                if self.classifier_score(output1.best_seg) > self.classifier_score(output2.best_seg):
                    best_output = output1
                else:
                    best_output = output2
                images[image1] = best_output
            except Exception as e:
                continue
        return images

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
            keep_keys = random.sample(images.keys(), int(len(images) * 0.01))
            logger.info(f"only keeping random 1% of {len(images)} input, i.e. {len(keep_keys)}")
            #keep_keys = random.sample(images.keys(), int(len(images) * 0.01))
            images = images.filter_by(keep_keys)

        # Load embeddings of segments
        if self.args.load_embs:
            logger.info("loading saved embedding")
            embs_featex = self.load_embeddings(self.args.save_root.parent / "embs" / self.target_class / "xception.pkl")
            if self.args.dataset == "coco":
                embs_cls = self.load_embeddings(self.args.save_root.parent / "embs" / self.target_class / "q2l.pkl")
            elif self.args.dataset == "voc":
                embs_cls = self.load_embeddings(self.args.save_root.parent / "embs" / self.target_class / "resnet.pkl")
            else:
                raise NotImplementedError
        else:
            embs_featex, embs_cls = None, None

        # Reomve person-like segments
        if self.args.ignore_person and embs_cls is not None:
            assert self.args.dataset == "coco", "able to get cls score from embedding only for coco"
            logger.info("removing person-like segments")
            for image_id, output in tqdm(images.items(), desc="removing person segments"):
                for m, m_id in output:
                    emb = self.retrieve_embedding(embs_cls, [image_id], [[m_id]])
                    if emb is None or not np.all(emb) or emb.size == 0:
                        continue
                    cls_score = torch.squeeze(torch.Tensor(emb)).softmax(-1)
                    if torch.numel(cls_score) != 80:
                        continue
                    if cls_score[0] > 0.3:
                        output.mask[output.mask == m_id] = 255

        # Load step1
        if self.args.load_step1 and os.path.exists(self.args.save_root.parent / "step-1" / self.target_class):
            self.load_step1(images, self.args.save_root.parent / "step-1")
        else:
            images = self.select_seg_by_gradcam(images)
            self.save(images, iter=-1, output_dir=self.args.save_root, filter_thresh=0.1)

        for iter in tqdm(range(self.args.num_iter), desc="EM iter"):
            k = self.args.M_k[min(iter, len(self.args.M_k)-1)] if type(self.args.M_k) is list else self.args.M_k
            _, mean_emb_fe = self.M_step(
                images.copy(),
                feat_extractor=self.feat_extractor,
                mode=self.args.M_mode,
                k=k,
                metric=self.args.M_metric,
                n_cluster=self.args.M_n_cluster,
                filter_thresh=self.args.filter_thresh,
                saved_embs=embs_featex
            )
            _, mean_emb_cls = self.M_step(
                images.copy(),
                feat_extractor=self.classifier,
                mode=self.args.M_mode,
                k=k,
                metric=self.args.M_metric,
                n_cluster=self.args.M_n_cluster,
                filter_thresh=self.args.filter_thresh,
                saved_embs=embs_cls
            )
            images_fe = self.E_step(images.copy(), mean_emb_fe, feat_extractor=self.feat_extractor, saved_embs=embs_featex)
            images_cls = self.E_step(images.copy(), mean_emb_cls, feat_extractor=self.classifier, saved_embs=embs_cls)
            images = self.post_process(images_fe, images_cls)
            self.save(images, iter, output_dir=self.args.save_root, filter_thresh=self.args.filter_thresh)
