# src/transforms.py
import torchvision.transforms as T
import random

class Compose(object):
    """여러 transform을 순차적으로 적용"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    """PIL 이미지를 PyTorch Tensor로 변환"""
    def __call__(self, image, target):
        image = T.functional.to_tensor(image)
        return image, target


class RandomHorizontalFlip(object):
    """일정 확률로 이미지와 bbox를 좌우 반전"""
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            width = image.shape[-1]
            image = T.functional.hflip(image)

            boxes = target["boxes"]
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            target["boxes"] = boxes

        return image, target


def get_transform(train: bool = True):
    """
    학습(train=True) 또는 평가(train=False) 모드에 맞게 transform 구성
    """
    transforms = [ToTensor()]
    if train:
        transforms.append(RandomHorizontalFlip(0.5))  # 50% 확률로 좌우 반전
    return Compose(transforms)
