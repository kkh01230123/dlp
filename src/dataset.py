import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET


class PotholeDataset(Dataset):
    def __init__(self, root, split_file, transforms=None):
        self.root = root
        self.transforms = transforms

        self.img_dir = os.path.join(root, "images")
        self.ann_dir = os.path.join(root, "annotations")

        # split 파일에서 이미지 리스트 가져오기
        with open(split_file, "r") as f:
            self.image_ids = [x.strip() for x in f.readlines()]

    def __len__(self):
        return len(self.image_ids)

    def _find_image(self, fname):
        """확장자 자동 탐색"""
        base = os.path.join(self.img_dir, fname)

        # fname에 확장자가 이미 포함돼 있고 파일이 존재하는 경우
        if os.path.exists(base):
            return base

        # 확장자가 없거나 잘못된 경우 자동 탐색
        for ext in [".png", ".jpg", ".jpeg", ".JPG", ".PNG"]:
            path = base + ext
            if os.path.exists(path):
                return path

        raise FileNotFoundError(f"No image found for: {fname}")


    def __getitem__(self, idx):
        image_id = self.image_ids[idx]  # ex) potholes248.png

        # 확장자 자동 탐색
        img_path = self._find_image(image_id)

        # annotation은 xml 고정
        ann_path = os.path.join(self.ann_dir, image_id.split('.')[0] + ".xml")

        img = Image.open(img_path).convert("RGB")

        boxes = []
        tree = ET.parse(ann_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            b = obj.find("bndbox")
            xmin = float(b.find("xmin").text)
            ymin = float(b.find("ymin").text)
            xmax = float(b.find("xmax").text)
            ymax = float(b.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])

        filtered_boxes = []
        for (xmin, ymin, xmax, ymax) in boxes:
            if xmax > xmin and ymax > ymin:
                filtered_boxes.append([xmin, ymin, xmax, ymax])
            else:
                print(f"[Warning] Invalid box removed in {image_id}: {[xmin, ymin, xmax, ymax]}")

        boxes = torch.as_tensor(filtered_boxes, dtype=torch.float32)
        labels = torch.ones((len(filtered_boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        # dataset.py 내부 __getitem__에서
        if self.transforms:
            img, target = self.transforms(img, target)


        return img, target



def collate_fn(batch):
    return tuple(zip(*batch))

