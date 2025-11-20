# src/transforms.py
import torchvision.transforms.functional as F
import torchvision.transforms as T
import random

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            _, h, w = image.shape
            image = F.hflip(image)

            boxes = target["boxes"]
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target["boxes"] = boxes
        return image, target


def get_train_transform():
    return Compose([
        ToTensor(),
        RandomHorizontalFlip(0.5),
    ])

def get_valid_transform():
    return Compose([
        ToTensor(),
    ])