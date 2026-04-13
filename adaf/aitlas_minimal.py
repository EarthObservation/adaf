"""Minimal subset of AiTLAS used by ADAF inference.

This module keeps only the model and transform utilities required by
`adaf_inference.py` and `adaf_utils.py`.
"""

import os
from types import SimpleNamespace

import numpy as np
import rasterio
import timm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision


NHIGH = 120


class ResizeV2:
    """Convert HWC image array to CHW float tensor."""

    def __call__(self, sample):
        image = np.asarray(sample)
        return torch.from_numpy(image.transpose(2, 0, 1)).float()


class Transpose:
    """Convert HWC image array to CHW float tensor."""

    def __call__(self, sample):
        image = np.asarray(sample)
        return torch.from_numpy(image.transpose(2, 0, 1)).float()


class _BaseInferenceModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        defaults = {
            "num_classes": 2,
            "use_cuda": True,
            "pretrained": True,
            "threshold": 0.5,
        }
        merged = {**defaults, **(config or {})}
        self.config = SimpleNamespace(**merged)

        device_name = "cuda:0" if self.config.use_cuda and torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)

    def prepare(self):
        self.model = self.model.to(self.device)

    def load_model(self, file_path):
        if not os.path.isfile(file_path):
            raise ValueError(f"No checkpoint found at {file_path}")

        checkpoint = torch.load(file_path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()


class FasterRCNN(_BaseInferenceModel):
    def __init__(self, config):
        super().__init__(config)

        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if self.config.pretrained else None
        )

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.config.num_classes)

    @staticmethod
    def _nms_predictions(outputs, threshold=0.2):
        keep = torchvision.ops.nms(outputs[0]["boxes"], outputs[0]["scores"], threshold)
        final_prediction = outputs[0]
        final_prediction["boxes"] = final_prediction["boxes"][keep]
        final_prediction["scores"] = final_prediction["scores"][keep]
        final_prediction["labels"] = final_prediction["labels"][keep]
        return final_prediction

    def detect_objects_v2(self, image=None, labels=None, data_transforms=None, description=None):
        del labels, description
        self.model.eval()

        if data_transforms:
            image = data_transforms(image)

        if torch.is_tensor(image):
            inputs = image.type(torch.float32).unsqueeze(0).to(self.device)
        else:
            inputs = torch.from_numpy(image).type(torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs)

        return self._nms_predictions(outputs)


class HRNetModule(nn.Module):
    def __init__(self, head: nn.Module, pretrained: bool = True, higher_res: bool = False):
        super().__init__()
        self.head = head
        self.backbone = timm.create_model("hrnet_w48", pretrained=pretrained)
        if higher_res:
            self.backbone.conv2.stride = (1, 1)

    def forward(self, x):
        inshape = x.shape[-2:]
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.conv2(x)
        x = self.backbone.bn2(x)
        x = self.backbone.act2(x)

        x = self.backbone.layer1(x)

        xl = [t(x) for t in self.backbone.transition1]
        yl = self.backbone.stage2(xl)

        xl = [t(yl[-1]) if not isinstance(t, nn.Identity) else yl[i] for i, t in enumerate(self.backbone.transition2)]
        yl = self.backbone.stage3(xl)

        xl = [t(yl[-1]) if not isinstance(t, nn.Identity) else yl[i] for i, t in enumerate(self.backbone.transition3)]
        yl = self.backbone.stage4(xl)

        return {
            "out": F.interpolate(self.head(x, yl), size=inshape, mode="bilinear", align_corners=False)
        }


class HRNetSegHead(nn.Module):
    def __init__(self, nclasses: int = 3, higher_res: bool = False):
        super().__init__()
        self.res_modifier = 2 if higher_res else 1
        self.projection = nn.Sequential(
            nn.Conv2d(976, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, nclasses, 1),
        )

    def forward(self, x, yl):
        mod = self.res_modifier
        low_level = torch.cat([F.interpolate(feat, (NHIGH * mod, NHIGH * mod)) for feat in [x, *yl]], 1)
        return self.projection(low_level)


class HRNet(_BaseInferenceModel):
    def __init__(self, config, higher_res=False):
        super().__init__(config)
        self.model = HRNetModule(
            HRNetSegHead(self.config.num_classes, higher_res),
            self.config.pretrained,
            higher_res,
        )

    def _get_predicted(self, outputs, threshold=None):
        predicted_probs = torch.sigmoid(outputs)
        predicted = (predicted_probs >= (threshold if threshold else self.config.threshold)).long()
        return predicted_probs, predicted

    def predict_masks_tiff_probs_binary(
        self,
        image_path=None,
        label=None,
        data_transforms=None,
        predictions_dir=None,
        description=None,
    ):
        del description
        with rasterio.open(image_path) as image_tiff:
            img_tiff_data = image_tiff.read()
            meta = image_tiff.meta.copy()

        if img_tiff_data.shape[0] == 1:
            image = np.repeat(img_tiff_data, 3, axis=0)
        else:
            image = img_tiff_data

        image = np.transpose(image, (1, 2, 0))

        if data_transforms:
            image = data_transforms(image)

        if torch.is_tensor(image):
            inputs = image.unsqueeze(0).to(self.device)
        else:
            inputs = torch.from_numpy(image).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)

        if isinstance(outputs, dict):
            outputs = outputs["out"]

        predicted_probs, _ = self._get_predicted(outputs)
        p = predicted_probs[0][1].detach().cpu().numpy()
        p = np.reshape(p, (1,) + p.shape)

        out_path = os.path.join(predictions_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_{label}_segmentation_mask_probs.tif")
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(p)
