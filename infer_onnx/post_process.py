import cv2
import numpy as np
from infer_onnx.utils import xywh2xyxy


def post_process(prediction, image_size, resized_size, conf_thres=0.5, iou_thres=0.5):
    """
    Filter boxes by confidence threshold and apply NMS
    """
    prediction = np.squeeze(prediction).T
    scores = np.max(prediction[:, 4:], axis=1)
    prediction = prediction[scores > conf_thres, :]
    scores = scores[scores > conf_thres]
    class_ids = np.argmax(prediction[:, 4:], axis=1)

    # Get bounding boxes and change format
    boxes = prediction[:, :4]
    boxes = xywh2xyxy(boxes)
    nms_result = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=conf_thres, nms_threshold=iou_thres)

    # Rescale box
    output = np.concatenate((np.expand_dims(class_ids[nms_result], axis=-1),
                             scale_boxes(image_size, boxes[nms_result], resized_size),
                             np.expand_dims(scores[nms_result], axis=-1)), axis=1)
    return output

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """Rescale boxes (xyxy) from img1_shape to img0_shape"""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2