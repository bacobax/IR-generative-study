from __future__ import annotations

import numpy as np

from src.evaluation.detection_metrics import (
    DetectionGroundTruth,
    DetectionPrediction,
    box_iou_matrix,
    evaluate_detections,
    nms_numpy,
)


def test_box_iou_matrix_identity() -> None:
    boxes = np.asarray([[0.1, 0.1, 0.4, 0.4]], dtype=np.float32)
    iou = box_iou_matrix(boxes, boxes)

    assert iou.shape == (1, 1)
    assert iou[0, 0] == 1.0


def test_nms_suppresses_overlapping_lower_score_box() -> None:
    boxes = np.asarray(
        [
            [0.1, 0.1, 0.5, 0.5],
            [0.12, 0.12, 0.52, 0.52],
            [0.7, 0.7, 0.9, 0.9],
        ],
        dtype=np.float32,
    )
    scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)

    keep = nms_numpy(boxes, scores, iou_threshold=0.5)

    assert keep.tolist() == [0, 2]


def test_evaluate_detections_perfect_ap() -> None:
    predictions = [
        DetectionPrediction(
            image_id="img_a",
            boxes_xyxy=np.asarray([[0.2, 0.2, 0.5, 0.5]], dtype=np.float32),
            scores=np.asarray([0.9], dtype=np.float32),
            class_ids=np.asarray([0], dtype=np.int32),
        )
    ]
    ground_truths = [
        DetectionGroundTruth(
            image_id="img_a",
            boxes_xyxy=np.asarray([[0.2, 0.2, 0.5, 0.5]], dtype=np.float32),
            class_ids=np.asarray([0], dtype=np.int32),
        )
    ]

    result = evaluate_detections(predictions=predictions, ground_truths=ground_truths, names={0: "person"})

    assert result["summary"]["map50"] == 1.0
    assert result["summary"]["precision"] == 1.0
    assert result["summary"]["recall"] == 1.0
