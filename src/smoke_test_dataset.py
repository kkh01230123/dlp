from dataset import PotholeDataset, collate_fn
from transforms import get_transform
from torch.utils.data import DataLoader

dataset = PotholeDataset(
    images_dir="data/images",
    annotations_dir="data/annotations",
    split_file="data/splits/train.txt",
    transforms=get_transform(train=True)
)

dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

for images, targets in dataloader:
    print(images[0].shape)
    print(targets[0])
    break
