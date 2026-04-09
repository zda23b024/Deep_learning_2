import os
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader."""

    def __init__(self, root_dir: str, split: str = "train", mask: bool = False):
        self.root_dir = root_dir
        self.split = split
        self.mask = mask

        self.image_dir = os.path.join(root_dir, "images")
        self.anno_dir = os.path.join(root_dir, "annotations", "xmls")
        self.trimap_dir = os.path.join(root_dir, "annotations", "trimaps")

        # Load image IDs and labels
        if split in ["train", "val"]:
            list_file = os.path.join(root_dir, "annotations", "trainval.txt")
        else:
            list_file = os.path.join(root_dir, "annotations", f"{split}.txt")

        with open(list_file, "r") as f:
            lines = f.readlines()
            self.image_ids = [line.strip().split(" ")[0] for line in lines]
            self.labels = [int(line.strip().split(" ")[1]) - 1 for line in lines]

        # Base transforms
        transform_list = [
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]

        # Compose with bbox support
        if self.mask:
            self.transform = A.Compose(
                transform_list,
                bbox_params=A.BboxParams(
                    format='pascal_voc',
                    label_fields=[]
                ),
                additional_targets={'mask': 'mask'}
            )
        else:
            self.transform = A.Compose(
                transform_list,
                bbox_params=A.BboxParams(
                    format='pascal_voc',
                    label_fields=[]
                )
            )

    def _load_bbox(self, xml_path):
        """Load bbox in Pascal VOC format: [xmin, ymin, xmax, ymax]"""
        if not os.path.exists(xml_path):
            return [0, 0, 1, 1]

        tree = ET.parse(xml_path)
        root = tree.getroot()

        bbox = root.find("object").find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        return [xmin, ymin, xmax, ymax]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        image_path = os.path.join(self.image_dir, image_id + ".jpg")
        xml_path = os.path.join(self.anno_dir, image_id + ".xml")

        # Load image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # Load label
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Load bbox (Pascal VOC format)
        bbox = self._load_bbox(xml_path)

        # Load mask if needed
        segmentation_mask = None
        if self.mask:
            mask_path = os.path.join(self.trimap_dir, image_id + ".png")
            mask = Image.open(mask_path).convert("L")
            mask_np = np.array(mask) - 1
            mask_np = np.clip(mask_np, 0, 2)
            segmentation_mask = mask_np

        # Apply transformations
        if self.mask:
            transformed = self.transform(
                image=image_np,
                mask=segmentation_mask,
                bboxes=[bbox]
            )
            image = transformed['image']
            bbox = transformed['bboxes'][0]
            mask_tensor = transformed['mask'].long()
        else:
            transformed = self.transform(
                image=image_np,
                bboxes=[bbox]
            )
            image = transformed['image']
            bbox = transformed['bboxes'][0]
            mask_tensor = torch.empty(0, dtype=torch.long)

        # Convert bbox → [cx, cy, w, h]
        xmin, ymin, xmax, ymax = bbox

        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin

        bbox = torch.tensor(
            [x_center, y_center, width, height],
            dtype=torch.float32
        )

        return image, label, bbox, mask_tensor
