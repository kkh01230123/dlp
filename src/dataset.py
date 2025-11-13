# src/dataset.py
import os
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from PIL import Image

class PotholeDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, split_file, transforms=None):
        with open(split_file) as f:
            self.image_ids = [x.strip() for x in f.readlines()]
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        # 확장자 자동 처리
        if "." in image_id:
            img_path = os.path.join(self.images_dir, image_id)
        else:
            img_path = os.path.join(self.images_dir, f"{image_id}.jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(self.images_dir, f"{image_id}.png")

        ann_path = os.path.join(self.annotations_dir, f"{os.path.splitext(image_id)[0]}.xml")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {img_path}")

        img = Image.open(img_path).convert("RGB")
        boxes, labels = self.parse_xml(ann_path)
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx])
        }
        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target

    def parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes, labels = [], []
        for obj in root.findall("object"):
            name = obj.find("name").text.lower().strip()
            if name != "pothole":
                continue
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)
        return boxes, labels


def collate_fn(batch):
    return tuple(zip(*batch))
