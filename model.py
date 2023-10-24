import os.path
from typing import Iterator, OrderedDict
import numpy as np
import timm
import sys
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from gradcam import GradCAM


class Model(nn.Module):
    def __init__(self, model_name: str, batch_size=8):
        super().__init__()
        self.gradcam: GradCAM
        if model_name == "resnet":
            self.model = models.resnet50()
            self.model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
            num_ftrs = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(num_ftrs, 20)
            self.model.load_state_dict(torch.load("pretrained_models/voc_multilabel/resnet50/model-2.pth"))
        elif model_name == "q2l":
            if os.path.basename(sys.argv[0]) == "select_mask.py":
                batch_size = 1
            from coco_cls.query2label import build_q2l
            def clean_state_dict(state_dict):
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if k[:7] == 'module.':
                        k = k[7:]  # remove `module.`
                    new_state_dict[k] = v
                return new_state_dict
            self.model = build_q2l()
            checkpoint = torch.load("pretrained_models/coco_multilabel/checkpoint.pkl")
            state_dict = clean_state_dict(checkpoint['state_dict'])
            self.model.load_state_dict(state_dict, strict=True)
            del checkpoint
            del state_dict
            torch.cuda.empty_cache()
        else:
            raise NotImplementedError
        self.find_layer_idx()
        self.batch_size = batch_size
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])

    @property
    def device(self):
        return next(self.parameters()).device

    def find_target_layer(self) -> str:
        """
        find last conv layer
        :return: layername
        """
        layer_name = None
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                layer_name = name
        if layer_name is None:
            raise ValueError("Could not find conv2d layer. Cannot apply GradCAM")
        return layer_name

    def forward(self, img):
        """
        :param img: RGB (H, W, 3) (int, unnormalized) or (N, H, W, 3)
        :return: pred and tensorized normalized img
        """
        if not torch.is_tensor(img):
            img = torch.Tensor(img)
        if img.dim() == 4:
            img = torch.permute(img, (0, 3, 1, 2)) / 255
        else:
            assert img.dim() == 3
            img = torch.permute(img, (2, 0, 1)) / 255
            img = img.unsqueeze(0)  # (1, 3, H, W)

        img = img.to(self.device)
        if os.path.basename(sys.argv[0]) == "save_embs.py":
            with torch.no_grad():
                preds = self.model(img)  # (N, K)
        else:
            preds = self.model(img)  # (N, K)
        return preds.softmax(-1), img

    def register_gradcam(self, layer_name, img_size):
        self.gradcam = GradCAM(self.model, layer_name, img_size)

    def compute_gradcam(self, img):
        """
        :param img: (H, W, 3) int8
        :return: GradCAM heatmap (H, W)
        """
        with torch.no_grad():
            # (1, K) preds
            preds, x = self(img)
            pred_class = preds[0].argmax()
        x.requires_grad_(True)
        cam = self.gradcam(x, pred_class)
        cam3 = np.expand_dims(cam, axis=2)
        cam3 = np.tile(cam3, [1, 1, 3])
        keep_percent = 20
        # (H, W, 3)
        cam = np.where(cam3 > np.percentile(cam3, 100 - keep_percent), 1, 0)
        # only need 1 channel
        return cam[..., 0]

    def find_layer_idx(self):
        self.layers = {}
        self.target_layer_idx = {}
        for idx, (name, module) in enumerate(self.model.named_modules()):
            self.target_layer_idx[name] = idx
            self.layers[idx] = module

    def compute_embedding(self, images):
        emb_layer = "avgpool" if "avgpool" in self.target_layer_idx.keys() else "fc"
        def preprocess_img(images):
            x = np.ascontiguousarray(images)
            x = x[np.newaxis, ...]
            x = torch.tensor(x, requires_grad=False, dtype=torch.float32)
            return x

        embs = []
        def hook(module, input, output):
            embs.append(output.clone().detach())

        if emb_layer == "fc":
            handle = self.model.fc.register_forward_hook(hook)
        else:
            handle = self.layers[self.target_layer_idx[emb_layer]].register_forward_hook(hook)
        segs = [
            preprocess_img(image)
            for image in images
        ]
        # (N, C=3, H, W)
        segs = torch.stack(segs, dim=0)
        for seg in segs.split(self.batch_size):
            # each seg (3, H, W)
            _ = self(seg.squeeze())
        handle.remove()
        # (N, d)
        return np.squeeze(np.concatenate([emb.detach().cpu() for emb in embs], axis=0))


class FeatureExtractor(nn.Module):
    def __init__(self, model_name: str, batch_size: int):
        super().__init__()
        assert model_name in timm.list_models(pretrained=True)
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, segs):
        # (N, C=3, H, W) -> (N, d)
        embs = self.model(segs)
        return embs.cpu().numpy()
    
    @torch.no_grad()
    def compute_embedding(self, images: Iterator[np.ndarray]) -> np.ndarray:
        segs = [
            self.transform(Image.fromarray(image))
            for image in images
        ]
        # (N, C=3, H, W)
        segs = torch.stack(segs, dim=0)
        embs = []
        for seg in segs.split(self.batch_size):
            # each seg (bsz, 3, H, W)
            emb = self(seg.to(self.device))
            embs.append(emb)
        # (N, d)
        return np.concatenate(embs, axis=0)
