# src/evaluation.py
import numpy as np

def compute_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_min + (y2_max - y2_min))  # safe

    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0

    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def match_predictions_to_gt(pred_boxes, gt_boxes, iou_thresh=0.5):
    # pred_boxes, gt_boxes: list[list[4]]
    matched_gt = set()
    tp = 0
    fp = 0

    for pb in pred_boxes:
        best_iou = 0.0
        best_idx = -1
        for gi, gb in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_idx = gi
        if best_iou >= iou_thresh and best_idx >= 0:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn

def eval_precision_recall_f1(predictions, ground_truths, iou_thresh=0.5):
    # predictions: list of dict {'boxes': [[...]], 'scores': [...]}
    # ground_truths: list of dict {'boxes': [[...]]}
    total_tp = total_fp = total_fn = 0
    for pred, gt in zip(predictions, ground_truths):
        tp, fp, fn = match_predictions_to_gt(pred['boxes'], gt['boxes'], iou_thresh)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall) else 0.0

    return {
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
        'precision': precision, 'recall': recall, 'f1': f1
    }

def eval_ap_single_threshold(predictions, ground_truths, iou_thresh=0.5):
    # 간단 AP: score로 정렬 후 커브 적분 (VOC 2007 style 근사)
    # 모든 예측을 score 기준으로 모아 전역 정렬
    pool = []
    total_gt = sum(len(gt['boxes']) for gt in ground_truths)
    for img_idx, pred in enumerate(predictions):
        for b, s in zip(pred['boxes'], pred['scores']):
            pool.append((s, img_idx, b))
    pool.sort(key=lambda x: x[0], reverse=True)

    matched = {i: set() for i in range(len(ground_truths))}
    tps, fps = [], []
    for s, img_idx, box in pool:
        gt_boxes = ground_truths[img_idx]['boxes']
        # 최대 IoU GT 찾기
        best_iou, best_idx = 0.0, -1
        for gi, gb in enumerate(gt_boxes):
            if gi in matched[img_idx]:
                continue
            iou = compute_iou(box, gb)
            if iou > best_iou:
                best_iou, best_idx = iou, gi
        if best_iou >= iou_thresh and best_idx >= 0:
            matched[img_idx].add(best_idx)
            tps.append(1)
            fps.append(0)
        else:
            tps.append(0)
            fps.append(1)

    if total_gt == 0:
        return {'ap': 0.0, 'precision': [], 'recall': []}

    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(fps)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    recall = cum_tp / total_gt

    # VOC style AP (11-point interpolation)
    ap = 0.0
    for r in np.linspace(0, 1, 11):
        p = precision[recall >= r].max() if np.any(recall >= r) else 0.0
        ap += p / 11.0

    return {'ap': float(ap), 'precision': precision.tolist(), 'recall': recall.tolist()}
