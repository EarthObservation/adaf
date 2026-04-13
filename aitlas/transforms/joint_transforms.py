import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2

from ..base import BaseTransforms


class FlipHVRandomRotate(BaseTransforms):
    def __call__(self, sample):
        image, mask = sample
        image = np.asarray(image)
        mask = np.asarray(mask)
        data_transforms = A.Compose(
            [
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomRotate90(),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.2,
                    rotate_limit=15,
                    p=0.9,
                    border_mode=cv2.BORDER_REFLECT,
                ),
            ]
        )
        transformed = data_transforms(image=image, mask=mask)

        return transformed["image"], transformed["mask"]


class FlipHVToTensorV2(BaseTransforms):
    def __call__(self, sample):
        image, target = sample
        data_transforms = A.Compose(
            [
                A.Resize(480, 480),
                A.HorizontalFlip(0.5),
                A.VerticalFlip(0.5),
                A.RandomRotate90(),
                ToTensorV2(p=1.0),
            ],
            bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
        )
        transformed = data_transforms(
            image=image, bboxes=target["boxes"], labels=target["labels"]
        )
        target["boxes"] = torch.Tensor(transformed["bboxes"])
        return transformed["image"], target

class FlipHVToTensorV3(BaseTransforms):
    def __call__(self, sample):
        image, target = sample
        data_transforms = A.Compose(
            [
                A.HorizontalFlip(0.5),
                A.VerticalFlip(0.5),
                A.RandomRotate90(),
				A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.2,
                    rotate_limit=15,
                    p=0.9,
                    border_mode=cv2.BORDER_REFLECT,
                ),
                ToTensorV2(p=1.0),
            ],
            bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
        )
        transformed = data_transforms(
            image=image, bboxes=target["boxes"], labels=target["labels"]
        )
        if len(transformed["bboxes"]) == 0 or len(transformed["bboxes"])!=len(target["boxes"]):
            # Bounding box is outside bounds of image, or one of the boxes is outside of bounds -> skip transformation
            transformed = ToTensorV2(p=1.0)(image=image)
            return transformed["image"], target
        else:
            # Return transformed image and bounding box
            target["boxes"] = torch.Tensor(transformed["bboxes"])
            return transformed["image"], target
        

class ResizeToTensorV2(BaseTransforms):
    def __call__(self, sample):
        image, target = sample
        data_transforms = A.Compose(
            [A.Resize(480, 480), ToTensorV2(p=1.0)],
            bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
        )

        transformed = data_transforms(
            image=image, bboxes=target["boxes"], labels=target["labels"]
        )
        target["boxes"] = torch.Tensor(transformed["bboxes"])

        return transformed["image"], target

class NoResizeToTensorV2(BaseTransforms):
    def __call__(self, sample):
        image, target = sample
        data_transforms = A.Compose([
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

        transformed = data_transforms(image=image, bboxes=target['boxes'], labels=target["labels"])
        target['boxes'] = torch.Tensor(transformed['bboxes'])

        return transformed["image"], target
    
class Resize(BaseTransforms):
    def __call__(self, sample):
        data_transforms = A.Compose([A.Resize(480, 480)])

        transformed = data_transforms(image=sample)

        return transformed["image"]

class ResizeV2(BaseTransforms):
    def __call__(self, sample):
        data_transforms = A.Compose(
            [ToTensorV2(p=1.0)])

        transformed = data_transforms(image=sample)

        return transformed["image"]