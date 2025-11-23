# src/test.py

import argparse
import os
import torch
from torch.utils.data import DataLoader

from src.dataset import PotholeDataset, collate_fn
from src.transforms import get_valid_transform
from src.model import get_faster_rcnn_model
from src.evaluation import eval_precision_recall_f1, eval_ap_single_threshold

from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image


# ----------------------------
# 디바이스 선택
# ----------------------------
def get_device(arg_device: str):
    if arg_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg_device


# ----------------------------
# Test DataLoader 구성
# ----------------------------
def build_test_loader(args, device):
    test_list = os.path.join(args.splits_dir, "test.txt") \
        if args.test_split == "" else args.test_split

    dataset = PotholeDataset(
        root=args.data_dir,
        split_file=test_list,
        transforms=get_valid_transform(),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    return loader


# ----------------------------
# Best 모델 로드
# ----------------------------
def load_model(args, device):
    model = get_faster_rcnn_model(num_classes=args.num_classes)

    ckpt_path = args.checkpoint
    if ckpt_path is None or ckpt_path == "":
        # models/<run_name>/best_epoch_x.pt 형태로 자동 구성
        ckpt_path = os.path.join(args.models_dir, args.run_name, args.ckpt_name)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model, ckpt_path


# ----------------------------
# 예측 결과 시각화 및 저장
# ----------------------------
def visualize_and_save(image_tensor, outputs, save_path, score_thresh=0.5):
    # image_tensor: [C,H,W], float(0~1)
    img_uint8 = (image_tensor.clamp(0, 1) * 255).to(torch.uint8)

    boxes = outputs["boxes"]
    labels = outputs.get("labels", None)
    scores = outputs["scores"]

    keep = scores >= score_thresh
    boxes = boxes[keep]
    scores = scores[keep]

    if boxes.numel() == 0:
        pil_img = to_pil_image(img_uint8)
        pil_img.save(save_path)
        return

    label_texts = [f"{s:.2f}" for s in scores.tolist()]

    drawn = draw_bounding_boxes(
        img_uint8,
        boxes,
        labels=label_texts,
        colors="red",
        width=2,
    )
    pil_img = to_pil_image(drawn)
    pil_img.save(save_path)


# ----------------------------
# 예측/정답 수집 (평가용)
# ----------------------------
def collect_predictions_and_gts(model, test_loader, device, score_thresh=0.5):
    """
    test 전체에 대해:
      - preds: [{'boxes': [[x1,y1,x2,y2], ...], 'scores': [..]}]
      - gts:   [{'boxes': [[...], ...]}]
    를 반환.
    """
    preds, gts = [], []
    model.eval()
    with torch.no_grad():
        for images, targets in test_loader:
            image = images[0].to(device)  # batch_size=1 가정
            output = model([image])[0]

            # 예측값
            boxes = output['boxes']
            scores = output['scores']
            keep = scores >= score_thresh
            pred_boxes = boxes[keep].cpu().numpy().tolist()
            pred_scores = scores[keep].cpu().numpy().tolist()
            preds.append({'boxes': pred_boxes, 'scores': pred_scores})

            # 정답값
            gt_boxes = targets[0]['boxes'].cpu().numpy().tolist()
            gts.append({'boxes': gt_boxes})
    return preds, gts


# ----------------------------
# 메인: Test + 시각화 + 평가
# ----------------------------
def main():
    parser = argparse.ArgumentParser()

    # 경로 관련
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--splits_dir", type=str, default="data/splits")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--outputs_dir", type=str, default="outputs")

    # 체크포인트 지정
    parser.add_argument("--run_name", type=str, default="FRcnn_bs4_lr0.00025_20251123-192443")
    parser.add_argument("--ckpt_name", type=str, default="best_epoch_16.pt")
    parser.add_argument("--checkpoint", type=str, default="", help="직접 경로 지정시 우선")

    # 모델/테스트 설정
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--score_thresh", type=float, default=0.8)

    # splits 직접 지정 옵션 (기본은 test.txt)
    parser.add_argument("--test_split", type=str, default="")

    args = parser.parse_args()

    device = get_device(args.device)
    print("Device:", device)

    # 출력 폴더 이름 결정
    run_name = args.run_name if args.run_name != "" else "test_run"
    out_dir = os.path.join(args.outputs_dir, run_name, "test_vis")
    os.makedirs(out_dir, exist_ok=True)

    # 모델 로드
    model, ckpt_path = load_model(args, device)
    print(f"Loaded checkpoint: {ckpt_path}")

    # DataLoader
    test_loader = build_test_loader(args, device)

    # Test inference + 시각화
    model.eval()
    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            image = images[0].to(device)  # batch_size=1 가정
            outputs = model([image])[0]

            save_name = f"test_{i:04d}.png"
            save_path = os.path.join(out_dir, save_name)

            visualize_and_save(
                image.cpu(),
                outputs,
                save_path,
                score_thresh=args.score_thresh,
            )

            if i < 5:
                print(f"[INFO] Saved: {save_path}")

    print("\n[Test] Inference & visualization finished.")
    print(f"Results saved in: {out_dir}")

    # ===== 여기부터: 예측 수집 + 평가 호출 =====
    preds, gts = collect_predictions_and_gts(
        model, test_loader, device, score_thresh=args.score_thresh
    )
    prf = eval_precision_recall_f1(preds, gts, iou_thresh=0.5)
    ap  = eval_ap_single_threshold(preds, gts, iou_thresh=0.5)

    # 결과 저장/출력
    eval_path = os.path.join(out_dir, "eval_summary.txt")
    with open(eval_path, "w") as f:
        f.write(f"ScoreThresh={args.score_thresh}\n")
        f.write(f"TP={prf['tp']} FP={prf['fp']} FN={prf['fn']}\n")
        f.write(f"Precision={prf['precision']:.4f} Recall={prf['recall']:.4f} F1={prf['f1']:.4f}\n")
        f.write(f"AP@0.5={ap['ap']:.4f}\n")

    print(f"[EVAL] P={prf['precision']:.4f} R={prf['recall']:.4f} F1={prf['f1']:.4f} | AP50={ap['ap']:.4f}")
    print(f"[EVAL] Saved: {eval_path}")


if __name__ == "__main__":
    main()

