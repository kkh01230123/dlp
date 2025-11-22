import argparse
import os
import json
import time
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from src.dataset import PotholeDataset, collate_fn
from src.transforms import get_train_transform, get_valid_transform
from src.model import get_faster_rcnn_model   # 너가 모델 만들면 여기에 연결


# -----------------------------------------------------------------------
#  SEED 고정
# -----------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# -----------------------------------------------------------------------
#  한 epoch 학습
# -----------------------------------------------------------------------
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0.0
    
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)            # dict
        loss = sum(loss_dict.values())                # tensor 하나로 합치기

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


# -----------------------------------------------------------------------
#  Validation은 mAP 계산이 복잡하므로 "loss 기반" 검증만 수행
# -----------------------------------------------------------------------
@torch.no_grad()
def validate(model, data_loader, device):
    model.train()  # eval() 대신 train()으로 두고, dropout/bn만 영향
    total_loss = 0.0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)      # 여기서 dict를 기대
        loss = sum(loss_dict.values())
        total_loss += loss.item()

    if len(data_loader) == 0:
        return 0.0
    return total_loss / len(data_loader)



# -----------------------------------------------------------------------
#  MAIN
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)

    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--splits_dir", default="data/splits")

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_dir", default="models")

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    # ----------------------------
    #  DEVICE 설정
    # ----------------------------
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Device: {device}")

    # ----------------------------
    #  RUN 디렉토리 생성
    # ----------------------------
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"FRcnn_bs{args.batch_size}_lr{args.lr}_{stamp}"
    run_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # config 저장
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # ----------------------------
    # Dataset 생성
    # ----------------------------
    train_list = os.path.join(args.splits_dir, "train.txt")
    val_list   = os.path.join(args.splits_dir, "val.txt")

    train_dataset = PotholeDataset(
        args.data_dir,
        train_list,
        transforms=get_train_transform()
    )

    valid_dataset = PotholeDataset(
        args.data_dir,
        val_list,
        transforms=get_valid_transform()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # ----------------------------
    #  MODEL 준비
    # ----------------------------
    model = get_faster_rcnn_model(num_classes=2)  # background + pothole(1)
    
    for name, param in model.backbone.named_parameters():
    # 예: layer2까지는 동결, layer3부터 학습
        if "layer3" not in name and "layer4" not in name:
            param.requires_grad = False
    
    model.to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005
    )

        # ----------------------------
    # 학습 루프 + Early Stopping
    # ----------------------------
    best_loss   = float("inf")
    best_path   = None

    patience    = 3          # 연속으로 개선이 없을 때 허용할 epoch 수
    no_improve  = 0          # 개선되지 않은 epoch 카운트

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        # 1) Train / Val 한 epoch 수행
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        val_loss   = validate(model, valid_loader, device)

        elapsed = time.time() - start

        print(
            f"[Epoch {epoch:03d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # 2) Best 모델 갱신 여부 확인
        if val_loss < best_loss:
            best_loss  = val_loss
            no_improve = 0  # 개선되었으니 카운터 리셋

            best_path = os.path.join(run_dir, f"best_epoch_{epoch}.pt")
            torch.save(model.state_dict(), best_path)
            print(f"[INFO] New best model at epoch {epoch} (val_loss={val_loss:.4f})")

        else:
            no_improve += 1
            print(f"[INFO] No improvement for {no_improve} epoch(s).")

            # 3) Early Stopping 조건 체크
            if no_improve >= patience:
                print(
                    f"[EARLY STOP] val_loss did not improve for {patience} "
                    f"consecutive epochs. Stop at epoch {epoch}."
                )
                break

    # 4) 마지막 epoch 모델도 별도 저장
    last_path = os.path.join(run_dir, "last.pt")
    torch.save(model.state_dict(), last_path)
    print(f"\nTraining completed. Best model saved to: {best_path}")
    print(f"Last model saved to: {last_path}")



if __name__ == "__main__":
    main()
