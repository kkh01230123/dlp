import os
import random
import glob

random.seed(42)

# 프로젝트 루트 기준 절대 경로 계산 (이 파일이 src/ 안에 있다고 가정)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE, 'data', 'images')
SPLIT_DIR = os.path.join(BASE, 'data', 'splits')
os.makedirs(SPLIT_DIR, exist_ok=True)

# 여러 확장자 지원
patterns = ['*.jpg', '*.jpeg', '*.png']  
img_paths = []
for pat in patterns:
    img_paths.extend(glob.glob(os.path.join(IMG_DIR, pat)))
img_paths = sorted(img_paths)

random.shuffle(img_paths)

n_total = len(img_paths)
n_train = int(n_total * 0.7)
n_val = int(n_total * 0.2)
n_test = n_total - n_train - n_val

train = img_paths[:n_train]
val   = img_paths[n_train:n_train + n_val]
test  = img_paths[n_train + n_val:]

def write_split(list_paths, out_path):
    with open(out_path, 'w') as f:
        for p in list_paths:
            f.write(os.path.basename(p) + '\n')

write_split(train, os.path.join(SPLIT_DIR, 'train.txt'))
write_split(val,   os.path.join(SPLIT_DIR, 'val.txt'))
write_split(test,  os.path.join(SPLIT_DIR, 'test.txt'))

print(f"총 이미지 개수: {n_total}")
print(f"Train: {len(train)}장, Val: {len(val)}장, Test: {len(test)}장")
